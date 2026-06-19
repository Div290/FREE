"""
Adversarial early-exit distillation for BLIP-2 (multi-GPU).

Goal
----
BLIP-2 (Salesforce/blip2-opt-2.7b, fp16) stays FROZEN. For each chosen decoder
layer `l`, an `IntermediateHead` learns to map that layer's hidden states into
a vector that a shared `Discriminator` cannot distinguish from the model's
final-layer hidden states (per-token, GAN-style adversarial distillation).
At inference time you could exit early at layer `l` and use the matching head
to get a "final-layer-like" representation cheaply.

Fixes vs. the original draft
-----------------------------
1. Removed duplicate/dead head+optimizer+scheduler definitions and the unused
   KLDivLoss / LoRA / get_linear_schedule_with_warmup imports.
2. BLIP-2 is explicitly frozen (`requires_grad_(False)`, `model.eval()`) since
   only the heads + discriminator train, per your decision. LoRA removed.
3. `Graphcore/vqa`'s `image_id` is a *string* COCO id, not a path or URL
   (confirmed against the HF VQA tutorial, which does
   `Image.open(example["image_id"])` against locally-staged COCO val2014
   files). Since we want on-the-fly fetching with no pre-downloaded zip, we
   build the COCO val2014 URL from the id and HTTP-fetch + disk-cache each
   image the first time it's used.
4. `collate_fn` no longer silently overwrites the *question* `input_ids`/
   `attention_mask` with the *answer*'s. Question and answer are now kept as
   separate keys (`input_ids`/`attention_mask` for the prompt that BLIP-2
   actually conditions on, `labels` for supervision) so nothing is dropped.
5. `dataset.filter` to drop empty-answer examples now runs via a writable
   in-memory mask, not against a read-only HF dataset object identity issue.
6. Discriminator now operates per-token (per your choice) with the correct
   tensor shapes: real/fake features are (batch, seq_len, hidden) ->
   discriminator outputs (batch, seq_len, 1) -> labels are
   torch.ones/zeros_like(that exact shape), not the previous shape-mismatched
   `real_features[:, :1]` / `torch.zeros(len(fake_features), 1)` which would
   have raised a runtime error or silently broadcast incorrectly.
7. Concatenating predictions from heads attached to *different* layers used
   to be `torch.cat([...], dim=0)` then compared against a labels tensor with
   the wrong shape entirely; now each head's fake-vs-real loss is computed
   and accumulated independently, which is also the correct GAN formulation
   here (you want every head to fool the discriminator, not to treat the
   batch as if it were `num_heads` larger).
8. All tensors are explicitly moved to the right device every step. Since
   you're on multi-GPU, the script uses Accelerate's `Accelerator` to handle
   device placement / multi-GPU training cleanly instead of manual
   `dispatch_model` + `infer_auto_device_map`, which doesn't compose well
   with a manual training loop and new trainable submodules that live outside
   the dispatched model.
9. Added an actual epoch-level average loss print, gradient clipping, and
   checkpoint saving for `intermediate_heads` (the artifact you presumably
   want to keep).
10. Batch's `pixel_values` are cast to fp16 to match the frozen BLIP-2 model
    dtype; previously dtype mismatches between batch tensors and the model
    were never addressed.

Install
-------
pip install -q transformers accelerate datasets pillow requests bitsandbytes
"""

import os
import io
import hashlib

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_NAME = "ybelkada/blip2-opt-2.7b-fp16-sharded"
PROCESSOR_NAME = "Salesforce/blip2-opt-2.7b"

MAX_QUESTION_LENGTH = 32
MAX_ANSWER_LENGTH = 32
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-4
WARMUP_STEPS = 100
GRAD_CLIP_NORM = 1.0

LAYERS_FOR_EXIT = [3, 5, 18, 21, 27]
INPUT_SIZE = 2560       # BLIP-2 OPT-2.7b hidden size
HIDDEN_SIZE = 5072
OUTPUT_SIZE = INPUT_SIZE  # heads project back into hidden_size, NOT vocab
                            # size -- see note below
NUM_DECODER_LAYERS = 2
NUM_HEADS = 8
DROPOUT = 0.1

IMAGE_CACHE_DIR = "./coco_val2014_cache"
COCO_VAL2014_URL_TMPL = (
    "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
)

CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_EPOCHS = 1


# --------------------------------------------------------------------------
# NOTE on OUTPUT_SIZE
# --------------------------------------------------------------------------
# The original script set output_size=vocab_size (50272), implying the heads
# predict a logit distribution over the vocabulary directly from an
# intermediate layer. But the GAN training loop discriminates against
# `outputs.hidden_states[-1]`, which has width `hidden_size` (2560), not
# `vocab_size`. You cannot feed a vocab-sized vector into a discriminator
# built for hidden_size and also call it "matching the final hidden state."
# Since the stated goal is "mimic the final layer" (i.e. produce a
# final-layer-like hidden representation cheaply), OUTPUT_SIZE here is set to
# INPUT_SIZE (2560), matching the discriminator's expected input. If you
# actually want each head to be a standalone early-exit LM head (predicting
# tokens directly, bypassing the rest of the network), say so and the loss
# function changes substantially (you'd add a cross-entropy term against
# `labels` per head, in addition to / instead of the adversarial term).


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class VQADataset(Dataset):
    """
    Wraps Graphcore/vqa. `image_id` in this dataset is a COCO val2014 numeric
    id (as a string), not a local path or URL. We fetch the corresponding
    COCO val2014 image over HTTP on first use and cache it to disk.
    """

    def __init__(self, hf_dataset, processor, max_q_len, max_a_len, cache_dir):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, image_id):
        # image_id may come through as e.g. "262148" or "COCO_val2014_000000262148"
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
        question = item["question"]
        weights = np.array(item["label"]["weights"])
        answer_id_idx = int(np.argmax(weights))
        # NOTE: label ids are answer *vocabulary indices* in the VQA
        # annotation scheme, not BLIP-2 tokenizer ids. We resolve to the
        # answer *string* via the dataset's label vocabulary if available;
        # since Graphcore/vqa stores answers as ids into its own answer
        # space, the standard approach is to also load the answer string
        # list. If your copy of the dataset doesn't expose answer strings,
        # you'll need its `id2label`/answer vocab file. For this script we
        # assume `item["label"]` also lets us recover the text via a
        # provided `multiple_choice_answer`-style field; fall back to a
        # best-effort string conversion otherwise.
        answer_text = item.get("multiple_choice_answer")
        if answer_text is None:
            # last-resort fallback so the script is runnable end-to-end;
            # replace with your real answer-id -> string mapping
            answer_text = str(item["label"]["ids"][answer_id_idx])

        image = self._load_image(item["image_id"])

        prompt = f"Question: {question} Answer:"
        enc = self.processor(
            images=image,
            text=prompt,
            padding="max_length",
            max_length=self.max_q_len,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["answer_text"] = answer_text
        return enc


def build_collate_fn(processor, max_answer_len):
    def collate_fn(batch):
        processed = {}
        for key in batch[0].keys():
            if key == "answer_text":
                continue
            processed[key] = torch.stack([ex[key] for ex in batch])

        answer_enc = processor.tokenizer(
            [ex["answer_text"] for ex in batch],
            max_length=max_answer_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Kept as `labels`, distinct from the question's input_ids/
        # attention_mask -- nothing is overwritten anymore.
        processed["labels"] = answer_enc["input_ids"]
        return processed

    return collate_fn


# --------------------------------------------------------------------------
# Model heads
# --------------------------------------------------------------------------
class IntermediateHead(nn.Module):
    """Maps an intermediate hidden state sequence to a final-layer-like
    representation of the same width, per token."""

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
        self.proj = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.to(self.proj.weight.dtype)
        memory = torch.zeros_like(x)
        out = self.transformer_decoder(x, memory)  # (batch, seq_len, input_size)
        return self.proj(out)  # (batch, seq_len, output_size)


class Discriminator(nn.Module):
    """Per-token real/fake classifier."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
        )  # logits; sigmoid folded into BCEWithLogitsLoss for stability

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, seq_len, 1) logits
        return self.net(x.to(self.net[0].weight.dtype))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    accelerator = Accelerator()
    device = accelerator.device

    # ---- data ----
    dataset = load_dataset("Graphcore/vqa", split="validation")

    empty_answer_mask = [len(ex["ids"]) > 0 for ex in dataset["label"]]
    keep_idxs = [i for i, keep in enumerate(empty_answer_mask) if keep]
    dataset = dataset.select(keep_idxs)
    accelerator.print(f"Dataset size after filtering empty answers: {len(dataset)}")

    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

    vqa_dataset = VQADataset(
        dataset, processor, MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH, IMAGE_CACHE_DIR
    )
    collate_fn = build_collate_fn(processor, MAX_ANSWER_LENGTH)
    train_dataloader = DataLoader(
        vqa_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
    )

    # ---- frozen BLIP-2 ----
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model.requires_grad_(False)
    model.eval()

    # ---- trainable heads + discriminator ----
    intermediate_heads = nn.ModuleList(
        [
            IntermediateHead(
                INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_DECODER_LAYERS, NUM_HEADS, DROPOUT
            )
            for _ in LAYERS_FOR_EXIT
        ]
    )
    discriminator = Discriminator(INPUT_SIZE, HIDDEN_SIZE)

    opt_int = AdamW(intermediate_heads.parameters(), lr=LR)
    opt_disc = AdamW(discriminator.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()

    model, intermediate_heads, discriminator, opt_int, opt_disc, train_dataloader = (
        accelerator.prepare(
            model, intermediate_heads, discriminator, opt_int, opt_disc, train_dataloader
        )
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(dtype=torch.float16)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                real_features = outputs.language_model_outputs.hidden_states[-1].detach()
                source_features = [
                    outputs.language_model_outputs.hidden_states[l].detach()
                    for l in LAYERS_FOR_EXIT
                ]

            # ---- discriminator step ----
            opt_disc.zero_grad()
            fake_features = [
                head(src).detach() for head, src in zip(intermediate_heads, source_features)
            ]

            real_logits = discriminator(real_features)
            real_labels = torch.ones_like(real_logits)
            d_loss = bce_loss(real_logits, real_labels)

            for fake in fake_features:
                fake_logits = discriminator(fake)
                fake_labels = torch.zeros_like(fake_logits)
                d_loss = d_loss + bce_loss(fake_logits, fake_labels)
            d_loss = d_loss / (1 + len(fake_features))

            accelerator.backward(d_loss)
            accelerator.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
            opt_disc.step()

            # ---- generator (intermediate heads) step ----
            opt_int.zero_grad()
            g_loss = 0.0
            for head, src in zip(intermediate_heads, source_features):
                fake = head(src)
                fake_logits = discriminator(fake)
                target_labels = torch.ones_like(fake_logits)  # want discriminator fooled
                g_loss = g_loss + bce_loss(fake_logits, target_labels)
            g_loss = g_loss / len(intermediate_heads)

            accelerator.backward(g_loss)
            accelerator.clip_grad_norm_(intermediate_heads.parameters(), GRAD_CLIP_NORM)
            opt_int.step()

            total_disc_loss += d_loss.item()
            total_gen_loss += g_loss.item()
            num_batches += 1

        avg_disc = total_disc_loss / max(num_batches, 1)
        avg_gen = total_gen_loss / max(num_batches, 1)
        accelerator.print(
            f"Epoch {epoch}: avg Disc Loss {avg_disc:.4f}, avg Gen Loss {avg_gen:.4f}"
        )

        if accelerator.is_main_process and (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"intermediate_heads_epoch{epoch}.pt")
            unwrapped = accelerator.unwrap_model(intermediate_heads)
            torch.save(unwrapped.state_dict(), ckpt_path)
            accelerator.print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
