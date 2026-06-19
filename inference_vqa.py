"""
Early Exit Inference — VQA (BLIP2-OPT-2.7b + LoRA + IntermediateHeads)
========================================================================
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


# ---------------------------------------------------------------------------
# Model definitions — must match training exactly
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.to(self.transformer_decoder_layer.self_attn.out_proj.weight.dtype)
        x = x.transpose(0, 1)                          # → (seq_len, batch, d)
        memory = torch.zeros_like(x)
        out = self.transformer_decoder(x, memory)
        out = out.transpose(0, 1)                      # → (batch, seq_len, d)
        logits = self.classifier(out.contiguous().view(-1, out.size(-1)))
        return logits.view(out.size(0), -1, logits.size(-1))  # (batch, seq_len, vocab)


class Discriminator(nn.Module):
    """Included for checkpoint compatibility; unused at inference."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture constants — must match training
# ---------------------------------------------------------------------------

LAYERS_FOR_EXIT = [3, 5, 18, 21, 27]
INPUT_SIZE       = 2560
HIDDEN_SIZE      = 5072
OUTPUT_SIZE      = 50272   # OPT-2.7b vocab size
NUM_LAYERS       = 2
NUM_HEADS        = 8
DROPOUT          = 0.1


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------

def confidence_max_prob(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability on the last token. Returns (B,)."""
    return F.softmax(logits[:, -1, :], dim=-1).max(dim=-1).values


def confidence_entropy(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """1 − normalised_entropy on the last token. Returns (B,)."""
    probs   = F.softmax(logits[:, -1, :], dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
    return 1.0 - entropy / torch.log(torch.tensor(float(vocab_size)))


def confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 probability on the last token. Returns (B,)."""
    probs = F.softmax(logits[:, -1, :], dim=-1)
    top2  = probs.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def get_confidence(logits: torch.Tensor, method: str, vocab_size: int) -> torch.Tensor:
    if method == "max_prob":
        return confidence_max_prob(logits)
    elif method == "entropy":
        return confidence_entropy(logits, vocab_size)
    elif method == "margin":
        return confidence_margin(logits)
    else:
        raise ValueError(f"Unknown confidence method: {method!r}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    backbone_id: str   = "ybelkada/blip2-opt-2.7b-fp16-sharded",
    processor_id: str  = "Salesforce/blip2-opt-2.7b",
    lora_ckpt: str | None   = None,
    heads_ckpt: str | None  = None,
    device: str        = "cuda",
):
    """
    Load backbone (+ optional LoRA) and intermediate heads.

    Parameters
    ----------
    backbone_id  : HuggingFace model id for the base BLIP2 weights.
    processor_id : HuggingFace id for the processor/tokeniser.
    lora_ckpt    : Path to a saved LoRA adapter directory
                   (e.g. from model.save_pretrained()).
                   Pass None to use the raw backbone.
    heads_ckpt   : Path to a .pt file saved as:
                       torch.save({"intermediate_heads": heads.state_dict()}, path)
                   Pass None for randomly-initialised heads (debug only).
    device       : Primary device; backbone is multi-GPU via device_map.

    Returns
    -------
    (processor, model, intermediate_heads)
    """
    print("[1/4] Loading processor …")
    processor = AutoProcessor.from_pretrained(processor_id)

    print("[2/4] Loading backbone …")
    backbone = Blip2ForConditionalGeneration.from_pretrained(
        backbone_id,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    )

    if lora_ckpt:
        print(f"      Applying LoRA weights from: {lora_ckpt}")
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", target_modules=["q_proj", "k_proj"],
        )
        backbone = get_peft_model(backbone, lora_cfg)
        backbone = PeftModel.from_pretrained(backbone, lora_ckpt)
    else:
        print("      No LoRA checkpoint — using base backbone.")

    print("[3/4] Dispatching backbone across available GPUs …")
    max_memory = get_balanced_memory(
        backbone,
        max_memory=None,
        no_split_module_classes=["Blip2ForConditionalGeneration", "IntermediateHead"],
        dtype="float16",
        low_zero=False,
    )
    device_map = infer_auto_device_map(
        backbone,
        max_memory=max_memory,
        no_split_module_classes=["Blip2ForConditionalGeneration", "IntermediateHead"],
        dtype="float16",
    )
    backbone = dispatch_model(backbone, device_map=device_map)
    backbone.eval()

    print("[4/4] Loading intermediate heads …")
    heads = nn.ModuleList([
        IntermediateHead(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
        for _ in LAYERS_FOR_EXIT
    ])

    if heads_ckpt:
        ckpt  = torch.load(heads_ckpt, map_location="cpu")
        state = ckpt.get("intermediate_heads", ckpt)
        heads.load_state_dict(state)
        print(f"      Loaded head weights from: {heads_ckpt}")
    else:
        print("      No head checkpoint — using random weights (debug only).")

    heads = heads.to(device).half().eval()

    return processor, backbone, heads


# ---------------------------------------------------------------------------
# Core early-exit inference  (single forward pass — suited for VQA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def early_exit_forward(
    model,
    intermediate_heads,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    confidence_threshold: float = 0.90,
    confidence_method: str      = "max_prob",
    layers_for_exit: list[int]  = LAYERS_FOR_EXIT,
    vocab_size: int             = OUTPUT_SIZE,
) -> dict:
    """
    Single forward pass with early-exit decision per sample in the batch.

    Unlike autoregressive tasks, VQA answers are typically one token, so we:
      • Run the full backbone once (to get all hidden states cheaply via cache).
      • Check each intermediate head's last-position logits against the threshold.
      • Assign each sample its own exit point independently (vectorised).

    Parameters
    ----------
    model               : dispatch_model'd BLIP2 backbone.
    intermediate_heads  : nn.ModuleList of IntermediateHead.
    pixel_values        : (B, C, H, W) fp16 tensor.
    input_ids           : (B, seq_len) — question tokens.
    attention_mask      : (B, seq_len).
    confidence_threshold: float in [0, 1].
    confidence_method   : "max_prob" | "entropy" | "margin".
    layers_for_exit     : list of OPT decoder layer indices to attach heads.
    vocab_size          : vocabulary size for normalised entropy.

    Returns
    -------
    dict with keys:
        "token_ids"      : (B,) LongTensor — predicted token id per sample.
        "exit_layers"    : (B,) list[int | str] — which layer each sample exited.
        "confidences"    : (B,) list[float] — confidence at exit point.
        "logits_final"   : (B, seq_len, vocab) — full-model logits (always computed).
    """
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # tuple: embedding + one per layer

    B              = pixel_values.size(0)
    chosen_tokens  = torch.full((B,), -1,    dtype=torch.long,  device=pixel_values.device)
    chosen_conf    = torch.zeros(B,           dtype=torch.float, device=pixel_values.device)
    exit_layers    = ["full"] * B
    exited_mask    = torch.zeros(B,           dtype=torch.bool,  device=pixel_values.device)

    for head, layer_idx in zip(intermediate_heads, layers_for_exit):
        if exited_mask.all():
            break  # every sample has already exited

        h      = hidden_states[layer_idx + 1]   # +1: index 0 is the embedding layer
        logits = head(h)                         # (B, seq_len, vocab)
        conf   = get_confidence(logits, confidence_method, vocab_size)  # (B,)

        # Samples that pass the threshold AND have not already exited
        fire_mask = (conf >= confidence_threshold) & (~exited_mask)

        if fire_mask.any():
            fired_tokens = logits[:, -1, :].argmax(dim=-1)  # (B,)
            chosen_tokens[fire_mask]  = fired_tokens[fire_mask]
            chosen_conf[fire_mask]    = conf[fire_mask]
            for i in fire_mask.nonzero(as_tuple=True)[0].tolist():
                exit_layers[i] = layer_idx
            exited_mask |= fire_mask

    # Fill remaining samples with full-model prediction
    still_running = ~exited_mask
    if still_running.any():
        full_logits = outputs.logits                              # (B, seq_len, vocab)
        full_tokens = full_logits[:, -1, :].argmax(dim=-1)       # (B,)
        full_conf   = confidence_max_prob(full_logits)
        chosen_tokens[still_running] = full_tokens[still_running]
        chosen_conf[still_running]   = full_conf[still_running]
        # exit_layers already "full" for these indices

    return {
        "token_ids"   : chosen_tokens,
        "exit_layers" : exit_layers,
        "confidences" : chosen_conf.tolist(),
        "logits_final": outputs.logits,
    }


# ---------------------------------------------------------------------------
# High-level predict function
# ---------------------------------------------------------------------------

def predict(
    image: Image.Image,
    question: str,
    processor,
    model,
    intermediate_heads,
    confidence_threshold: float = 0.90,
    confidence_method: str      = "max_prob",
    max_length: int             = 32,
    device: str                 = "cuda",
) -> dict:
    """
    Run early-exit inference on a single (image, question) pair.

    Parameters
    ----------
    image               : PIL.Image — the VQA image.
    question            : str — the question.
    processor           : BLIP2 processor.
    model               : dispatch_model'd BLIP2 backbone.
    intermediate_heads  : nn.ModuleList of IntermediateHead.
    confidence_threshold: float — exit if any head confidence ≥ this.
    confidence_method   : "max_prob" | "entropy" | "margin".
    max_length          : max token length for the question prompt.
    device              : tensor device.

    Returns
    -------
    dict:
        "answer"      : str  — decoded answer text
        "exit_layer"  : int | "full" — which layer provided the answer
        "confidence"  : float
    """
    encoding = processor(
        image, question,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    pixel_values  = encoding["pixel_values"].to(device).half()
    input_ids     = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    result = early_exit_forward(
        model=model,
        intermediate_heads=intermediate_heads,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        confidence_threshold=confidence_threshold,
        confidence_method=confidence_method,
    )

    answer = processor.tokenizer.decode(
        result["token_ids"], skip_special_tokens=True
    ).strip()

    return {
        "answer"    : answer,
        "exit_layer": result["exit_layers"][0],
        "confidence": result["confidences"][0],
    }


# ---------------------------------------------------------------------------
# Batch inference with per-sample exit tracking
# ---------------------------------------------------------------------------

def batch_predict(
    samples: list[dict],
    processor,
    model,
    intermediate_heads,
    confidence_threshold: float = 0.90,
    confidence_method: str      = "max_prob",
    batch_size: int             = 16,
    max_length: int             = 32,
    device: str                 = "cuda",
) -> list[dict]:
    """
    Run early-exit inference over a list of VQA samples.

    Each sample dict must have:
        "image"    : PIL.Image
        "question" : str
        "answer"   : str  (optional ground-truth, used for accuracy reporting)

    Returns
    -------
    List of dicts, one per sample, each with:
        "predicted"   : str
        "ground_truth": str | None
        "exit_layer"  : int | "full"
        "confidence"  : float
        "correct"     : bool | None
    """
    results   = []
    n_batches = (len(samples) + batch_size - 1) // batch_size

    for b in range(n_batches):
        chunk = samples[b * batch_size : (b + 1) * batch_size]

        encodings = [
            processor(
                s["image"], s["question"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for s in chunk
        ]

        pixel_values   = torch.cat([e["pixel_values"]   for e in encodings], dim=0).to(device).half()
        input_ids      = torch.cat([e["input_ids"]      for e in encodings], dim=0).to(device)
        attention_mask = torch.cat([e["attention_mask"] for e in encodings], dim=0).to(device)

        with torch.no_grad():
            out = early_exit_forward(
                model=model,
                intermediate_heads=intermediate_heads,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                confidence_threshold=confidence_threshold,
                confidence_method=confidence_method,
            )

        for i, sample in enumerate(chunk):
            predicted = processor.tokenizer.decode(
                [out["token_ids"][i].item()], skip_special_tokens=True
            ).strip()
            gt      = sample.get("answer", None)
            correct = (predicted.lower() == gt.lower()) if gt is not None else None

            results.append({
                "predicted"   : predicted,
                "ground_truth": gt,
                "exit_layer"  : out["exit_layers"][i],
                "confidence"  : out["confidences"][i],
                "correct"     : correct,
            })

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            done = min((b + 1) * batch_size, len(samples))
            print(f"  [{done}/{len(samples)}] processed")

    # Summary statistics
    exit_counts = {}
    for r in results:
        k = str(r["exit_layer"])
        exit_counts[k] = exit_counts.get(k, 0) + 1

    correct_results = [r for r in results if r["correct"] is not None]
    if correct_results:
        accuracy = sum(r["correct"] for r in correct_results) / len(correct_results)
        print(f"\nAccuracy : {accuracy:.4f}  ({sum(r['correct'] for r in correct_results)}/{len(correct_results)})")

    print("Exit layer distribution:")
    for layer, count in sorted(exit_counts.items(), key=lambda x: x[0]):
        pct = 100 * count / len(results)
        print(f"  Layer {layer:>4} : {count:>5} samples  ({pct:.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Graphcore/vqa dataset loader helper
# ---------------------------------------------------------------------------

def load_graphcore_vqa_samples(split: str = "validation") -> list[dict]:
    """
    Load Graphcore/vqa split into the flat dict format expected by batch_predict.

    Mirrors the training code's label extraction:
        answer = label["ids"][argmax(label["weights"])]
    """
    from datasets import load_dataset as hf_load

    print(f"Loading Graphcore/vqa ({split}) …")
    raw = hf_load("Graphcore/vqa", split=split)

    # Filter out samples with no labels (same as training)
    raw = raw.filter(lambda ex, idx: len(ex["label"]["ids"]) > 0, with_indices=True)

    samples = []
    for item in raw:
        weights = np.array(item["label"]["weights"])
        answer  = item["label"]["ids"][np.argmax(weights)]
        image   = Image.open(item["image_id"]).convert("RGB")
        samples.append({
            "image"   : image,
            "question": item["question"],
            "answer"  : answer,
        })

    print(f"  {len(samples)} samples loaded.")
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Early-exit BLIP2 VQA inference")
    p.add_argument("--image",     type=str, required=True,
                   help="Path to the input image.")
    p.add_argument("--question",  type=str, required=True,
                   help="The VQA question.")
    p.add_argument("--lora_ckpt", type=str, default=None,
                   help="Path to LoRA adapter directory.")
    p.add_argument("--heads_ckpt",type=str, default=None,
                   help="Path to heads checkpoint (.pt).")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Confidence threshold for early exit (default 0.90).")
    p.add_argument("--method",    type=str, default="max_prob",
                   choices=["max_prob", "entropy", "margin"],
                   help="Confidence estimation method.")
    p.add_argument("--eval",      action="store_true",
                   help="Run batch eval on Graphcore/vqa validation split.")
    p.add_argument("--device",    type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    processor, model, heads = load_models(
        lora_ckpt=args.lora_ckpt,
        heads_ckpt=args.heads_ckpt,
        device=args.device,
    )

    if args.eval:
        samples = load_graphcore_vqa_samples(split="validation")
        batch_predict(
            samples=samples,
            processor=processor,
            model=model,
            intermediate_heads=heads,
            confidence_threshold=args.threshold,
            confidence_method=args.method,
            device=args.device,
        )
    else:
        image  = Image.open(args.image).convert("RGB")
        result = predict(
            image=image,
            question=args.question,
            processor=processor,
            model=model,
            intermediate_heads=heads,
            confidence_threshold=args.threshold,
            confidence_method=args.method,
            device=args.device,
        )
        print("\n=== Result ===")
        print(f"Answer     : {result['answer']}")
        print(f"Exit layer : {result['exit_layer']}")
        print(f"Confidence : {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
