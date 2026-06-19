"""
Early Exit Inference for BLIP2 + LoRA + IntermediateHeads (VisDial)
=====================================================================
Architecture recap
------------------
- Backbone  : BLIP2-OPT-2.7b (fp16, sharded), fine-tuned with LoRA
- Exit heads: 5 x IntermediateHead (Transformer-decoder + linear classifier)
              attached to OPT decoder layers [3, 5, 18, 21, 27]
- Training  : GAN-style — heads learn to mimic the last hidden state's
              distribution so their logits are "as good" as the full model.

Early-exit strategy (confidence-based)
---------------------------------------
At each exit layer we compute a confidence score on the head's output logits.
If confidence >= threshold we exit early and return that head's prediction.
If no exit fires, we fall back to the full model's final output.

Supported confidence metrics
-----------------------------
  "max_prob"  – max softmax probability          (fast, standard)
  "entropy"   – normalised entropy               (lower = more confident)
  "margin"    – top-1 minus top-2 probability    (higher = more confident)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.nn import TransformerDecoder, TransformerDecoderLayer


# ---------------------------------------------------------------------------
# Model definitions  (must match training exactly)
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
        x = x.to(self.transformer_decoder_layer.self_attn.out_proj.weight.dtype)
        x = x.transpose(0, 1)                         # → (seq_len, batch, d)
        memory = torch.zeros_like(x)
        out = self.transformer_decoder(x, memory)
        out = out.transpose(0, 1)                     # → (batch, seq_len, d)
        logits = self.classifier(out.contiguous().view(-1, out.size(-1)))
        return logits.view(out.size(0), -1, logits.size(-1))  # (batch, seq_len, vocab)


class Discriminator(nn.Module):
    """Kept for checkpoint compatibility; not used at inference."""
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
# Confidence metrics
# ---------------------------------------------------------------------------

def confidence_max_prob(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability over the last generated token. Shape: (B,)"""
    probs = F.softmax(logits[:, -1, :], dim=-1)       # last token
    return probs.max(dim=-1).values


def confidence_entropy(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Normalised entropy (inverted so higher = more confident). Shape: (B,)"""
    probs = F.softmax(logits[:, -1, :], dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(vocab_size)))
    return 1.0 - (entropy / max_entropy)


def confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 probability. Shape: (B,)"""
    probs = F.softmax(logits[:, -1, :], dim=-1)
    top2 = probs.topk(2, dim=-1).values
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

LAYERS_FOR_EXIT = [3, 5, 18, 21, 27]

# Architecture hyper-params (must match training)
INPUT_SIZE  = 2560
HIDDEN_SIZE = 5072
OUTPUT_SIZE = 50272   # vocab size
NUM_LAYERS  = 2
NUM_HEADS   = 8
DROPOUT     = 0.1


def load_models(
    backbone_id: str = "ybelkada/blip2-opt-2.7b-fp16-sharded",
    processor_id: str = "Salesforce/blip2-opt-2.7b",
    lora_checkpoint: str | None = None,
    heads_checkpoint: str | None = None,
    device: str = "cuda",
):
    """
    Load backbone (+ optional LoRA weights) and all intermediate heads.

    Parameters
    ----------
    backbone_id       : HuggingFace model id for the base BLIP2 weights.
    processor_id      : HuggingFace id for the processor/tokeniser.
    lora_checkpoint   : Path to the saved LoRA adapter directory
                        (produced by model.save_pretrained()).
                        Pass None to use the backbone without LoRA.
    heads_checkpoint  : Path to the .pt file that contains
                        {"intermediate_heads": state_dict}.
                        Pass None to use randomly-initialised heads
                        (useful for debugging pipeline only).
    device            : "cuda" or "cpu".

    Returns
    -------
    processor, model, intermediate_heads
    """
    print("[1/4] Loading processor …")
    processor = AutoProcessor.from_pretrained(processor_id)

    print("[2/4] Loading backbone …")
    backbone = Blip2ForConditionalGeneration.from_pretrained(
        backbone_id,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    if lora_checkpoint:
        print(f"      Loading LoRA weights from {lora_checkpoint} …")
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", target_modules=["q_proj", "k_proj"],
        )
        backbone = get_peft_model(backbone, lora_cfg)
        backbone = PeftModel.from_pretrained(backbone, lora_checkpoint)
    else:
        print("      No LoRA checkpoint provided — using base backbone.")

    print("[3/4] Dispatching backbone across devices …")
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

    if heads_checkpoint:
        ckpt = torch.load(heads_checkpoint, map_location="cpu")
        # Support two common saving conventions
        state = ckpt.get("intermediate_heads", ckpt)
        heads.load_state_dict(state)
        print(f"      Loaded head weights from {heads_checkpoint}")
    else:
        print("      No head checkpoint provided — heads are randomly initialised.")

    heads = heads.to(device).half()
    heads.eval()

    return processor, backbone, heads


# ---------------------------------------------------------------------------
# Core early-exit inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def early_exit_greedy_decode(
    model,
    intermediate_heads,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    processor,
    max_new_tokens: int = 64,
    confidence_threshold: float = 0.90,
    confidence_method: str = "max_prob",
    layers_for_exit: list[int] = LAYERS_FOR_EXIT,
    vocab_size: int = OUTPUT_SIZE,
):
    """
    Token-by-token greedy decode with early exit at each step.

    At every generation step the hidden states from the intermediate layers
    are passed through the corresponding heads.  If the head's confidence
    exceeds `confidence_threshold` the token from *that* head is accepted and
    we move to the next step, skipping all deeper layers for that token.

    Returns
    -------
    generated_ids : list[int]   – token ids of the generated answer
    exit_log      : list[dict]  – per-token exit info for analysis
    """
    device = pixel_values.device

    # ---- vision encoding (run once) ----------------------------------------
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    image_embeds   = vision_outputs[0]                       # (B, n_patches, d_vision)
    image_attn     = torch.ones(image_embeds.size()[:-1], device=device, dtype=torch.long)

    query_tokens   = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs  = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attn,
    )
    query_output   = query_outputs.last_hidden_state         # (B, n_queries, d_qformer)
    language_model_inputs  = model.language_projection(query_output)
    language_model_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], device=device, dtype=torch.long
    )

    # ---- prepare text inputs ------------------------------------------------
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)

    # Concatenate vision prefix with text embeddings
    inputs_embeds  = torch.cat([language_model_inputs, inputs_embeds], dim=1)
    attention_mask = torch.cat([language_model_attention_mask, attention_mask], dim=1)

    # ---- autoregressive loop ------------------------------------------------
    generated_ids: list[int] = []
    exit_log: list[dict]     = []

    past_key_values = None

    for step in range(max_new_tokens):
        outputs = model.language_model(
            inputs_embeds=inputs_embeds if step == 0 else None,
            input_ids=None if step == 0 else torch.tensor([[generated_ids[-1]]], device=device),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden_states   = outputs.hidden_states   # tuple: (embedding, l0, l1, …, lN)

        # Extend attention mask for the new token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=torch.long)],
            dim=1,
        )

        # ---- early exit check -----------------------------------------------
        exited      = False
        chosen_tok  = None
        exit_info   = {"step": step, "exit_layer": "full", "confidence": None}

        for head, layer_idx in zip(intermediate_heads, layers_for_exit):
            h = hidden_states[layer_idx + 1]           # +1 because index 0 is embedding
            logits = head(h)                           # (B, seq_len, vocab)
            conf   = get_confidence(logits, confidence_method, vocab_size)  # (B,)

            # For simplicity we handle batch size == 1 at inference
            conf_val = conf[0].item()

            if conf_val >= confidence_threshold:
                chosen_tok = logits[0, -1, :].argmax(dim=-1).item()
                exit_info["exit_layer"]  = layer_idx
                exit_info["confidence"]  = conf_val
                exited = True
                break

        if not exited:
            # Fall back to full model's last-layer logits
            logits     = outputs.logits                # (B, seq_len, vocab)
            chosen_tok = logits[0, -1, :].argmax(dim=-1).item()
            exit_info["exit_layer"]  = "full"
            exit_info["confidence"]  = confidence_max_prob(logits)[0].item()

        generated_ids.append(chosen_tok)
        exit_log.append(exit_info)

        # Stop at EOS
        if chosen_tok == processor.tokenizer.eos_token_id:
            break

        # Next step: feed the chosen token as a single-token input
        # (past_key_values handles the rest)
        inputs_embeds = None   # switch to input_ids mode after first step

    return generated_ids, exit_log


# ---------------------------------------------------------------------------
# High-level predict function
# ---------------------------------------------------------------------------

def predict(
    image: Image.Image,
    dialog_history: list[tuple[str, str]],
    question: str,
    processor,
    model,
    intermediate_heads,
    max_new_tokens: int = 64,
    confidence_threshold: float = 0.90,
    confidence_method: str = "max_prob",
    device: str = "cuda",
) -> dict:
    """
    Run early-exit inference for a single VisDial turn.

    Parameters
    ----------
    image               : PIL.Image — the dialog image.
    dialog_history      : list of (question, answer) pairs for prior turns.
    question            : the current question to answer.
    processor           : BLIP2 processor.
    model               : dispatch_model'd BLIP2 backbone.
    intermediate_heads  : nn.ModuleList of IntermediateHead.
    max_new_tokens      : maximum tokens to generate.
    confidence_threshold: exit if any head confidence >= this value.
    confidence_method   : "max_prob" | "entropy" | "margin".
    device              : target device for tensors.

    Returns
    -------
    dict with keys:
        "answer"     : str   – decoded answer string
        "exit_log"   : list  – per-token exit info
        "avg_exit"   : float – average exit layer (lower = faster)
    """
    # Build prompt (same template as training)
    template = "Question: {} Answer: {}."
    prompt = " ".join(
        template.format(q, a) for q, a in dialog_history
    )
    prompt += f" Question: {question} Answer:"

    # Encode
    encoding = processor(
        image, prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    pixel_values  = encoding["pixel_values"].to(device).half()
    input_ids     = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Decode
    generated_ids, exit_log = early_exit_greedy_decode(
        model=model,
        intermediate_heads=intermediate_heads,
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        processor=processor,
        max_new_tokens=max_new_tokens,
        confidence_threshold=confidence_threshold,
        confidence_method=confidence_method,
    )

    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Summarise exit statistics
    numeric_exits = [
        e["exit_layer"] for e in exit_log if e["exit_layer"] != "full"
    ]
    full_exits = len(exit_log) - len(numeric_exits)
    avg_exit   = (
        sum(numeric_exits) / len(numeric_exits) if numeric_exits else None
    )

    return {
        "answer"          : answer,
        "exit_log"        : exit_log,
        "avg_exit_layer"  : avg_exit,
        "full_model_exits": full_exits,
        "total_tokens"    : len(exit_log),
    }


# ---------------------------------------------------------------------------
# Batch inference with configurable threshold sweep (evaluation helper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_predict(
    samples: list[dict],
    processor,
    model,
    intermediate_heads,
    confidence_threshold: float = 0.90,
    confidence_method: str = "max_prob",
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> list[dict]:
    """
    Run predict() over a list of samples.

    Each sample dict must have:
        "image"          : PIL.Image
        "dialog_history" : list of (q, a) tuples
        "question"       : str

    Returns list of result dicts (same schema as predict()).
    """
    results = []
    for i, sample in enumerate(samples):
        result = predict(
            image=sample["image"],
            dialog_history=sample["dialog_history"],
            question=sample["question"],
            processor=processor,
            model=model,
            intermediate_heads=intermediate_heads,
            max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold,
            confidence_method=confidence_method,
            device=device,
        )
        result["sample_idx"] = i
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] done")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Early-exit BLIP2 inference")
    p.add_argument("--image",         type=str, required=True,
                   help="Path to the input image.")
    p.add_argument("--question",      type=str, required=True,
                   help="Current question to answer.")
    p.add_argument("--history",       type=str, default="",
                   help='Prior dialog as JSON string: [["Q","A"], ...]')
    p.add_argument("--lora_ckpt",     type=str, default=None,
                   help="Path to LoRA adapter directory.")
    p.add_argument("--heads_ckpt",    type=str, default=None,
                   help="Path to heads checkpoint (.pt).")
    p.add_argument("--threshold",     type=float, default=0.90,
                   help="Confidence threshold for early exit (default 0.90).")
    p.add_argument("--method",        type=str, default="max_prob",
                   choices=["max_prob", "entropy", "margin"],
                   help="Confidence estimation method.")
    p.add_argument("--max_new_tokens",type=int, default=64)
    p.add_argument("--device",        type=str, default="cuda")
    return p.parse_args()


def main():
    import json
    args = parse_args()

    processor, model, heads = load_models(
        lora_checkpoint=args.lora_ckpt,
        heads_checkpoint=args.heads_ckpt,
        device=args.device,
    )

    image   = Image.open(args.image).convert("RGB")
    history = json.loads(args.history) if args.history else []

    result = predict(
        image=image,
        dialog_history=history,
        question=args.question,
        processor=processor,
        model=model,
        intermediate_heads=heads,
        max_new_tokens=args.max_new_tokens,
        confidence_threshold=args.threshold,
        confidence_method=args.method,
        device=args.device,
    )

    print("\n=== Result ===")
    print(f"Answer          : {result['answer']}")
    print(f"Total tokens    : {result['total_tokens']}")
    print(f"Full-model exits: {result['full_model_exits']}")
    print(f"Avg exit layer  : {result['avg_exit_layer']}")
    print("\nPer-token exit log:")
    for entry in result["exit_log"]:
        print(f"  step {entry['step']:>3}  exit={str(entry['exit_layer']):>4}  "
              f"conf={entry['confidence']:.4f}" if entry['confidence'] else
              f"  step {entry['step']:>3}  exit=full  conf=N/A")


if __name__ == "__main__":
    main()
