"""
Early-exit intermediate LM heads for BLIP-2 (Salesforce/blip2-flan-t5-xl).

Install
-------
pip install -q transformers accelerate datasets pillow requests sentencepiece
"""

import os

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets import load_dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

MAX_LENGTH = 10  # matches your draft's answer max length
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
WARMUP_STEPS = 100
SAVE_STEPS = 3000

LAYERS_FOR_EXIT = [3, 5, 7, 9, 12, 15, 18, 21, 23]  # valid: flan-t5-xl has 24 decoder layers (0-23)
INPUT_SIZE = 2048    # T5-XL d_model
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 32128  # T5 vocab size
NUM_DECODER_LAYERS = 2
NUM_HEADS = 8
DROPOUT = 0.1

IMAGE_CACHE_DIR = "./coco_val2014_cache"
COCO_VAL2014_URL_TMPL = "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"

CHECKPOINT_DIR = "./multi_heads_vqa_enc_dec/checkpoint/intermediate_head_weights"
LOG_FILE = "generated_vqa_enc_dec.txt"


# --------------------------------------------------------------------------
# Dataset: image-only input, answer text as label (per your confirmed intent)
# --------------------------------------------------------------------------
class VQAImageOnlyDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length, cache_dir):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, image_id):
        digits = "".join(ch for ch in str(image_id) if ch.isdigit())
        numeric_id = int(digits)
        cache_path = os.path.join(self.cache_dir, f"{numeric_id:012d}.jpg")
        if not os.path.exists(cache_path):
            url = COCO_VAL2014_URL_TMPL.format(image_id=numeric_id)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(resp.content)
        return Image.open(cache_path).convert("RGB")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        weights = np.array(item["label"]["weights"])
        answer_id_idx = int(np.argmax(weights))

        answer_text = item.get("multiple_choice_answer")
        if answer_text is None:
            # Fallback only -- replace with your real VQA answer-id -> string
            # vocabulary if `multiple_choice_answer` isn't present in your
            # copy of the dataset.
            answer_text = str(item["label"]["ids"][answer_id_idx])

        image = self._load_image(item["image_id"])

        # image-only encoding: no question text passed to the processor
        encoding = self.processor(images=image, return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["answer_text"] = answer_text
        return encoding


def build_collate_fn(processor, max_length):
    pad_token_id = processor.tokenizer.pad_token_id

    def collate_fn(batch):
        processed = {}
        for key in batch[0].keys():
            if key == "answer_text":
                continue
            processed[key] = torch.stack([ex[key] for ex in batch])

        answer_enc = processor.tokenizer(
            [ex["answer_text"] for ex in batch],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = answer_enc["input_ids"]
        # Mask padding so it's excluded from the loss (T5 pads with
        # pad_token_id, not -100, by default).
        labels = labels.masked_fill(labels == pad_token_id, -100)
        processed["labels"] = labels
        return processed

    return collate_fn


# --------------------------------------------------------------------------
# Intermediate head with real cross-attention to encoder memory
# --------------------------------------------------------------------------
class IntermediateHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x, memory):
        """
        x:      (batch, tgt_seq_len, input_size)  -- intermediate decoder hidden states
        memory: (batch, src_seq_len, input_size)  -- real T5 encoder output (image features)
        """
        dtype = self.classifier.weight.dtype
        x = x.to(dtype)
        memory = memory.to(dtype)
        out = self.transformer_decoder(tgt=x, memory=memory)  # (batch, tgt_seq_len, input_size)
        return self.classifier(out)  # (batch, tgt_seq_len, output_size)


def main():
    dataset = load_dataset("Graphcore/vqa", split="validation")

    keep_mask = [len(ex["ids"]) > 0 for ex in dataset["label"]]
    dataset = dataset.select([i for i, keep in enumerate(keep_mask) if keep])
    print("The length of dataset is", len(dataset))

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    model.requires_grad_(False)
    model.eval()

    vqa_dataset = VQAImageOnlyDataset(dataset, processor, MAX_LENGTH, IMAGE_CACHE_DIR)
    collate_fn = build_collate_fn(processor, MAX_LENGTH)
    train_dataloader = DataLoader(
        vqa_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=4
    )

    intermediate_heads = nn.ModuleList(
        [
            IntermediateHead(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_DECODER_LAYERS, NUM_HEADS, DROPOUT)
            for _ in LAYERS_FOR_EXIT
        ]
    )
    # Heads run on a single device; with model sharded via device_map="auto"
    # this keeps head training simple. Pick the device holding the LM head /
    # last decoder shard so cross-device copies stay minimal.
    heads_device = next(model.parameters()).device
    intermediate_heads = intermediate_heads.to(heads_device)

    optimizer = AdamW(intermediate_heads.parameters(), lr=LR)
    n_train_steps = NUM_EPOCHS * len(train_dataloader)
    print("The number of training steps are:", n_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=n_train_steps
    )

    input_device = model.get_input_embeddings().weight.device
    current_step = 0
    intermediate_heads.train()

    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0

        for idx, batch in enumerate(train_dataloader):
            print("Samples Processed:", idx)
            pixel_values = batch["pixel_values"].to(input_device, torch.float16)
            labels = batch["labels"].to(input_device)

            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                )
                encoder_memory = outputs.language_model_outputs.encoder_last_hidden_state.detach()
                decoder_hidden_states = outputs.language_model_outputs.decoder_hidden_states

            labels_for_loss = labels.to(heads_device)
            memory_for_heads = encoder_memory.to(heads_device)

            int_loss_train = 0.0
            for exit_idx, layer in enumerate(LAYERS_FOR_EXIT):
                head = intermediate_heads[exit_idx]
                hidden_state = decoder_hidden_states[layer].to(heads_device)

                intermediate_logits = head(hidden_state, memory_for_heads)

                if current_step > 0 and current_step % SAVE_STEPS == 0:
                    generated_caption = processor.batch_decode(
                        intermediate_logits.argmax(dim=-1), skip_special_tokens=True
                    )[0]
                    with open(LOG_FILE, "a") as f:
                        f.write(f"Exit: {layer}, Caption: {generated_caption}\n")

                reshaped_logits = intermediate_logits.reshape(-1, intermediate_logits.size(-1))
                intermediate_loss = F.cross_entropy(
                    reshaped_logits, labels_for_loss.reshape(-1), ignore_index=-100
                )
                int_loss_train = int_loss_train + layer * intermediate_loss

            int_loss_train = int_loss_train / len(LAYERS_FOR_EXIT)

            int_loss_train.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_v = int_loss_train.item()
            train_loss += loss_v
            current_step += 1

            if current_step % SAVE_STEPS == 0:
                print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / SAVE_STEPS:.4f}")
                train_loss = 0.0

                ckpt_dir = os.path.join(CHECKPOINT_DIR, f"-{current_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                for layer_idx, head in enumerate(intermediate_heads):
                    head_path = os.path.join(ckpt_dir, f"head_layer_{LAYERS_FOR_EXIT[layer_idx]}.pt")
                    torch.save(head.state_dict(), head_path)


if __name__ == "__main__":
    main()
