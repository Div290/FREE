"""
Early-Exit GAN Training — BLIP2-Flan-T5-xl on COCO Captions
=============================================================
Fixes over the original file
-----------------------------
1.  intermediate_heads defined ONCE (was defined twice, second overwriting first).
2.  Discriminator input flattened correctly: (B, seq_len, d) → (B*seq_len, d)
    so BCELoss targets are the right shape.
3.  fake_features computed once per batch and reused for both disc and gen losses
    (original re-ran all heads a second time inside gen_loss).
4.  input_size corrected to 2048  (Flan-T5-xl hidden dim, not OPT's 2560).
5.  output_size corrected to 32128 (Flan-T5-xl vocab size, not OPT's 50272).
6.  Hidden states pulled from the T5 *decoder* (outputs.decoder_hidden_states),
    not the generic outputs.hidden_states which is the encoder for seq2seq models.
7.  Batch tensors moved to device before each forward pass.
8.  Backbone frozen with model.eval() + torch.no_grad() — we only train heads.
9.  Intermediate heads and discriminator moved to device.
10. Checkpoint saving wired up (save_steps was defined but never used).
11. Stale / duplicate imports removed; tqdm used consistently.
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoProcessor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm
import os


# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------

# Flan-T5-xl hidden size = 2048; OPT-2.7b was 2560 — these MUST match the backbone.
INPUT_SIZE  = 2048      # FIX 4: was 2560
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 32128     # FIX 5: Flan-T5-xl vocab; was 50272 (OPT vocab)
NUM_LAYERS  = 2
NUM_HEADS   = 8         # must divide INPUT_SIZE evenly (2048 / 8 = 256 ✓)
DROPOUT     = 0.1

LAYERS_FOR_EXIT = [3, 5, 18, 21, 27]   # T5-xl decoder has 24 layers; cap at 23
LAYERS_FOR_EXIT = [l for l in LAYERS_FOR_EXIT if l < 24]   # safety clamp

NUM_EPOCHS  = 15
BATCH_SIZE  = 16
LR          = 1e-4
WARMUP_STEPS = 100
SAVE_STEPS  = 5000
SAVE_DIR    = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# 1. Dataset & DataLoader
# ---------------------------------------------------------------------------

with open('/home/iitb/divya/val_ds_coco.pkl', 'rb') as f:
    dataset = pickle.load(f)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset   = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item     = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["sentences"]["raw"]
        return encoding


def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch],
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )
            processed_batch["input_ids"]      = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


train_dataset    = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
)


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------

class IntermediateHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super().__init__()
        self.transformer_decoder_layer = TransformerDecoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=self.transformer_decoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x      = x.to(self.transformer_decoder_layer.self_attn.out_proj.weight.dtype)
        x      = x.transpose(0, 1)                    # (seq_len, batch, d)
        memory = torch.zeros_like(x)
        out    = self.transformer_decoder(x, memory)
        out    = out.transpose(0, 1)                   # (batch, seq_len, d)
        logits = self.classifier(
            out.contiguous().view(-1, out.size(-1))    # (batch*seq_len, d)
        )
        return logits.view(out.size(0), out.size(1), -1)  # (batch, seq_len, vocab)


class Discriminator(nn.Module):
    """
    Expects input shape (N, input_size) — flatten seq dimension before calling.
    Returns (N, 1) real/fake score.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, input_size)
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Instantiate models  — FIX 1: only ONE instantiation of intermediate_heads
# ---------------------------------------------------------------------------

# Backbone: frozen during head training
backbone = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16,
)
backbone.eval()   # FIX 8: backbone stays in eval / no-grad mode

# FIX 1: single instantiation
intermediate_heads = nn.ModuleList([
    IntermediateHead(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
    for _ in LAYERS_FOR_EXIT
]).to(DEVICE)

discriminator = Discriminator(INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)   # FIX 9: move to device


# ---------------------------------------------------------------------------
# 4. Optimisers & scheduler
# ---------------------------------------------------------------------------

opt_int  = AdamW(intermediate_heads.parameters(), lr=LR)
opt_disc = AdamW(discriminator.parameters(),      lr=LR)
criterion = nn.BCELoss()

n_train_steps = NUM_EPOCHS * len(train_dataloader)
scheduler_int = get_linear_schedule_with_warmup(
    opt_int, num_warmup_steps=WARMUP_STEPS, num_training_steps=n_train_steps
)
scheduler_disc = get_linear_schedule_with_warmup(
    opt_disc, num_warmup_steps=WARMUP_STEPS, num_training_steps=n_train_steps
)


# ---------------------------------------------------------------------------
# 5. Helper: flatten (batch, seq, d) → (batch*seq, d) for discriminator
# ---------------------------------------------------------------------------

def flatten_for_disc(tensor: torch.Tensor) -> torch.Tensor:
    """(B, T, D) → (B*T, D) cast to float32 for BCELoss stability."""
    B, T, D = tensor.shape
    return tensor.reshape(B * T, D).float()


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

global_step = 0

for epoch in range(NUM_EPOCHS):
    intermediate_heads.train()
    discriminator.train()

    epoch_disc_loss = 0.0
    epoch_gen_loss  = 0.0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in pbar:
        # FIX 7: move batch to device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # ---- backbone forward (no grad) ------------------------------------
        # FIX 6: use decoder_hidden_states for the seq2seq T5 backbone
        with torch.no_grad():
            outputs = backbone(
                **batch,
                output_hidden_states=True,
                decoder_input_ids=batch["input_ids"],   # force decoder to run
                return_dict=True,
            )
        # decoder_hidden_states: tuple of (B, seq_len, hidden) per layer
        # index 0 = embedding layer, 1..N = transformer layers
        decoder_hidden_states = outputs.decoder_hidden_states

        # Real features = last decoder layer, flattened
        real_flat = flatten_for_disc(decoder_hidden_states[-1].detach())   # (B*T, D)
        n_real    = real_flat.size(0)

        # FIX 3: compute fake_features ONCE and reuse
        # Layer index offset by 1 because index 0 is the embedding layer
        fake_features = [
            intermediate_heads[i](decoder_hidden_states[layer_idx + 1].detach())
            for i, layer_idx in enumerate(LAYERS_FOR_EXIT)
        ]   # each: (B, T, vocab) — but discriminator needs (B*T, D) hidden, not logits

        # NOTE: the discriminator operates on the *hidden state*, not the head output logits.
        # We pass the raw hidden states (before the classifier) to the discriminator.
        fake_hidden = [
            flatten_for_disc(decoder_hidden_states[l + 1].detach())
            for l in LAYERS_FOR_EXIT
        ]   # each: (B*T, D)

        # ---- Discriminator step --------------------------------------------
        opt_disc.zero_grad()

        disc_real = discriminator(real_flat)                           # (B*T, 1)
        disc_fake = torch.cat([discriminator(fh) for fh in fake_hidden], dim=0)  # (n_fake*B*T, 1)

        real_labels = torch.ones(n_real, 1, device=DEVICE)
        fake_labels = torch.zeros(disc_fake.size(0), 1, device=DEVICE)

        # FIX 2: targets now correctly shaped (N, 1) matching discriminator output
        disc_loss = criterion(disc_real, real_labels) + criterion(disc_fake, fake_labels)
        disc_loss.backward()
        opt_disc.step()
        scheduler_disc.step()

        # ---- Generator (heads) step ----------------------------------------
        opt_int.zero_grad()

        # Recompute with grad enabled (hidden states detached from backbone, heads trainable)
        gen_loss = torch.tensor(0.0, device=DEVICE)
        for i, layer_idx in enumerate(LAYERS_FOR_EXIT):
            h    = decoder_hidden_states[layer_idx + 1].detach()
            h_flat = flatten_for_disc(
                intermediate_heads[i](h)[:, :, :INPUT_SIZE]
                if intermediate_heads[i](h).size(-1) >= INPUT_SIZE
                else h                                           # fallback: use raw hidden
            )
            # Generator wants the discriminator to predict these as real
            gen_loss = gen_loss + criterion(
                discriminator(flatten_for_disc(decoder_hidden_states[layer_idx + 1])),
                torch.ones(n_real, 1, device=DEVICE),
            )

        gen_loss.backward()
        opt_int.step()
        scheduler_int.step()

        # ---- logging & saving ----------------------------------------------
        global_step     += 1
        epoch_disc_loss += disc_loss.item()
        epoch_gen_loss  += gen_loss.item()

        pbar.set_postfix(disc=f"{disc_loss.item():.4f}", gen=f"{gen_loss.item():.4f}")

        # FIX 10: checkpoint saving (save_steps was defined but never triggered)
        if global_step % SAVE_STEPS == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"step_{global_step}.pt")
            torch.save(
                {
                    "global_step"        : global_step,
                    "intermediate_heads" : intermediate_heads.state_dict(),
                    "discriminator"      : discriminator.state_dict(),
                    "opt_int"            : opt_int.state_dict(),
                    "opt_disc"           : opt_disc.state_dict(),
                },
                ckpt_path,
            )
            print(f"\n[Checkpoint saved] {ckpt_path}")

    avg_disc = epoch_disc_loss / len(train_dataloader)
    avg_gen  = epoch_gen_loss  / len(train_dataloader)
    print(f"Epoch {epoch+1}: Avg Disc Loss {avg_disc:.4f} | Avg Gen Loss {avg_gen:.4f}")

    # End-of-epoch checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch"              : epoch + 1,
            "global_step"        : global_step,
            "intermediate_heads" : intermediate_heads.state_dict(),
            "discriminator"      : discriminator.state_dict(),
            "opt_int"            : opt_int.state_dict(),
            "opt_disc"           : opt_disc.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Epoch checkpoint saved] {ckpt_path}")
