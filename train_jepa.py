"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default JEPA baseline run:
# - token+hash embedding dim 128, causal conv encoder to dim 480
# - learned chunk boundaries with hard-threshold STE boundary gate
# - 6 unique global blocks cycled twice (12 effective layers), dim 480
# - 2-layer decoder transformer at dim 256 with cross-attention to globals
# - CE token loss + training-time JEPA next-chunk prediction loss


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_pure_byte_260.json")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    val_sliding_stride = int(os.environ.get("VAL_SLIDING_STRIDE", 256))
    val_sliding_batch = int(os.environ.get("VAL_SLIDING_BATCH", 32))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 2))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32_768))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    model_dim = int(os.environ.get("MODEL_DIM", 480))
    encoder_token_dim = int(os.environ.get("ENCODER_TOKEN_DIM", 128))
    encoder_kernel_size = int(os.environ.get("ENCODER_KERNEL_SIZE", 4))
    hash_buckets = int(os.environ.get("HASH_BUCKETS", 2048))
    num_unique_global_blocks = int(os.environ.get("NUM_UNIQUE_GLOBAL_BLOCKS", 5))
    num_global_cycles = int(os.environ.get("NUM_GLOBAL_CYCLES", 2))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    decoder_dim = int(os.environ.get("DECODER_DIM", 480))
    decoder_layers = int(os.environ.get("DECODER_LAYERS", 7))
    decoder_heads = int(os.environ.get("DECODER_HEADS", 8))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))

    # Boundary predictor / JEPA.
    boundary_threshold = float(os.environ.get("BOUNDARY_THRESHOLD", 0.45))
    boundary_temperature = float(os.environ.get("BOUNDARY_TEMPERATURE", 12.0))
    jepa_weight = float(os.environ.get("JEPA_WEIGHT", 0.10))
    sigreg_weight = float(os.environ.get("SIGREG_WEIGHT", 0.10))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.30))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.


def build_pure_byte_luts(vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(vocab_size, 260)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    base_bytes_np[4 : min(table_size, 260)] = 1
    return (
        torch.tensor(base_bytes_np[:vocab_size], dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np[:vocab_size], dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np[:vocab_size], dtype=torch.bool, device=device),
    )


def load_pure_byte_luts(tokenizer_path: str, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    path = Path(tokenizer_path)
    if path.suffix != ".json":
        raise ValueError(f"Pure-byte JEPA expects a tokenizer JSON at {tokenizer_path!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_type = payload.get("tokenizer_type") or payload.get("kind")
    json_vocab_size = int(payload.get("vocab_size", vocab_size))
    if tokenizer_type != "pure_byte":
        raise ValueError(
            f"Unsupported tokenizer JSON {tokenizer_path}: expected pure_byte, got {tokenizer_type!r}"
        )
    if json_vocab_size != vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={json_vocab_size}"
        )
    return build_pure_byte_luts(vocab_size, device)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log_fn=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    stride = args.val_sliding_stride
    batch_seqs = args.val_sliding_batch
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= seq_len
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log_fn(
        f"ttt:start chunks={num_chunks} chunk_tokens={ttt_chunk} windows={len(window_starts)} "
        f"stride={stride} lr={args.ttt_lr} epochs={args.ttt_epochs}"
    )

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    t_infer_total = 0.0
    t_adapt_total = 0.0

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                for i, ws in enumerate(batch_ws):
                    chunk_tok = val_tokens[ws : ws + seq_len + 1].to(dtype=torch.int64, device=device)
                    x_batch[i] = chunk_tok[:-1]
                    y_batch[i] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    s = 0 if ws == 0 else max(seq_len - stride, 0)
                    scored = nll[i, s:seq_len].to(torch.float64)
                    loss_sum += scored.sum()
                    token_count += float(seq_len - s)
                    tgt = y_batch[i, s:seq_len]
                    prev = x_batch[i, s:seq_len]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        torch.cuda.synchronize()
        t_infer_total += time.perf_counter() - t_infer

        is_last_chunk = ci == num_chunks - 1
        t_adapt = time.perf_counter()
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                for _ep in range(args.ttt_epochs):
                    for bs in range(my_seq_s, my_seq_e, batch_seqs):
                        be = min(bs + batch_seqs, my_seq_e)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = base_model.forward_logits(x)
                        ce = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            y.reshape(-1),
                        )
                        ce.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        if args.ttt_grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        torch.cuda.synchronize()
        t_adapt_total += time.perf_counter() - t_adapt

        if rank == 0 and (ci % 50 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = (
                rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
                if token_count.item() > 0
                else 0.0
            )
            log_fn(
                f"ttt_chunk [{ci + 1}/{num_chunks}] bpb={rbpb:.6f} "
                f"time={elapsed:.1f}s infer={t_infer_total:.1f}s adapt={t_adapt_total:.1f}s"
            )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    base_model.eval()
    return val_loss, val_bpb


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# We quantize "middle" global blocks to int6 and all other large float tensors to int8,
# then compress with LZMA preset 9 to target a tighter submission size.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,boundary_threshold",
    ).split(",")
    if pattern
)
INTX_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INTX_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INTX_KEEP_FLOAT_MAX_NUMEL = 65_536
INTX_KEEP_FLOAT_STORE_DTYPE = torch.float16
INTX_PER_ROW_SCALE_DTYPE = torch.float16
INTX_CLIP_PERCENTILE = 99.99984
INTX_CLIP_Q = INTX_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INTX_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INTX_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def choose_quant_bits(name: str) -> int:
    # "Middle" unique global blocks (1..4) use int6. Everything else defaults to int8.
    if name.startswith("global_transformer.blocks."):
        parts = name.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            idx = int(parts[2])
            if 1 <= idx <= 4:
                return 6
    return 8


def quantize_float_tensor(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    max_q = (1 << (bits - 1)) - 1
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INTX_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / max_q).clamp_min(1.0 / max_q)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_q, max_q).to(torch.int8).contiguous()
        return q, scale.to(dtype=INTX_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INTX_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_q if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_q, max_q).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_intx(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    bitwidths: dict[str, int] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "intx_payload_bytes",
            "num_int6_tensors",
            "num_int8_tensors",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["intx_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INTX_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["intx_payload_bytes"] += tensor_nbytes(kept)
            continue

        bits = choose_quant_bits(name)
        stats["num_float_tensors"] += 1
        if bits == 6:
            stats["num_int6_tensors"] += 1
        else:
            stats["num_int8_tensors"] += 1
        q, s = quantize_float_tensor(t, bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        bitwidths[name] = bits
        stats["intx_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int6_8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "bitwidths": bitwidths,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_intx(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    bitwidths: dict[str, int] = obj.get("bitwidths", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        bits = int(bitwidths.get(name, 8))
        _ = (1 << (bits - 1)) - 1  # kept for explicitness of per-tensor quant range
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, num_heads: int):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q = CastedLinear(query_dim, query_dim, bias=False)
        self.k = CastedLinear(kv_dim, query_dim, bias=False)
        self.v = CastedLinear(kv_dim, query_dim, bias=False)
        self.proj = CastedLinear(query_dim, query_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        bsz, t_q, d_q = x.shape
        t_k = memory.size(1)
        q = self.q(x).reshape(bsz, t_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(memory).reshape(bsz, t_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(memory).reshape(bsz, t_k, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        attn_bias = torch.where(
            memory_mask[:, None, :, :],
            torch.zeros((), device=x.device, dtype=q.dtype),
            torch.full((), -1e4, device=x.device, dtype=q.dtype),
        )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        y = y.transpose(1, 2).contiguous().reshape(bsz, t_q, d_q)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, memory_dim: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.self_norm = RMSNorm()
        self.cross_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.self_attn = CausalSelfAttention(dim, num_heads, num_heads, rope_base, qk_gain_init)
        self.cross_attn = CrossAttention(dim, memory_dim, num_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.self_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.cross_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        x = x + self.self_scale.to(dtype=x.dtype)[None, None, :] * self.self_attn(self.self_norm(x))
        x = x + self.cross_scale.to(dtype=x.dtype)[None, None, :] * self.cross_attn(self.cross_norm(x), memory, memory_mask)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class HashNgramEmbedding(nn.Module):
    def __init__(self, hash_buckets: int, dim: int):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.trigram = nn.Embedding(hash_buckets, dim)
        self.fourgram = nn.Embedding(hash_buckets, dim)
        self.register_buffer("tri_coeffs", torch.tensor([1, 257, 65537], dtype=torch.int64), persistent=False)
        self.register_buffer("four_coeffs", torch.tensor([1, 257, 65537, 9973], dtype=torch.int64), persistent=False)

    def _hash_ngram(self, input_ids: Tensor, coeffs: Tensor) -> Tensor:
        n = int(coeffs.numel())
        padded = F.pad(input_ids, (n - 1, 0), value=0)
        windows = padded.unfold(1, n, 1).to(torch.int64)
        hashed = torch.remainder((windows * coeffs[None, None, :]).sum(dim=-1), self.hash_buckets)
        return hashed

    def forward(self, input_ids: Tensor) -> Tensor:
        tri_ids = self._hash_ngram(input_ids, self.tri_coeffs)
        four_ids = self._hash_ngram(input_ids, self.four_coeffs)
        return self.trigram(tri_ids) + self.fourgram(four_ids)


def ste_hard_threshold(prob: Tensor) -> Tensor:
    hard = (prob > 0.5).to(prob.dtype)
    return hard + prob - prob.detach()


def chunk_reduce(x: Tensor, boundary_hard: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Build per-token chunk IDs from hard boundaries and average token features per chunk.
    bsz, seqlen, dim = x.shape
    chunk_ids = torch.cumsum(boundary_hard.to(torch.int64), dim=1) - 1
    num_chunks = chunk_ids[:, -1] + 1
    max_chunks = int(num_chunks.max().item())
    chunk_latents = x.new_zeros((bsz, max_chunks, dim))
    chunk_counts = x.new_zeros((bsz, max_chunks, 1))
    ones = x.new_ones((seqlen, 1))
    for b in range(bsz):
        ids = chunk_ids[b]
        chunk_latents[b].index_add_(0, ids, x[b])
        chunk_counts[b].index_add_(0, ids, ones)
    chunk_latents = chunk_latents / chunk_counts.clamp_min_(1.0)
    valid_chunk_mask = torch.arange(max_chunks, device=x.device)[None, :] < num_chunks[:, None]
    return chunk_latents, chunk_ids, valid_chunk_mask


def sigreg_loss(z: Tensor) -> Tensor:
    if z.numel() == 0:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


class JEPAByteModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {args.logit_softcap}")
        if args.encoder_kernel_size < 2:
            raise ValueError("ENCODER_KERNEL_SIZE must be >= 2")

        self.args = args
        self.logit_softcap = args.logit_softcap
        self.boundary_temperature = args.boundary_temperature
        self.jepa_weight = args.jepa_weight
        self.sigreg_weight = args.sigreg_weight

        # Encoder: token emb + trigram/4-gram hash emb -> causal conv (128 -> 480).
        self.tok_emb = nn.Embedding(args.vocab_size, args.encoder_token_dim)
        self.hash_emb = HashNgramEmbedding(args.hash_buckets, args.encoder_token_dim)
        self.encoder_conv = nn.Conv1d(
            in_channels=args.encoder_token_dim,
            out_channels=args.model_dim,
            kernel_size=args.encoder_kernel_size,
            stride=1,
            bias=False,
        )
        self.encoder_norm = RMSNorm()
        self.boundary_threshold = nn.Parameter(torch.tensor(args.boundary_threshold, dtype=torch.float32))

        # Global transformer: 6 unique blocks reused for 2 cycles.
        self.global_transformer = nn.ModuleDict(
            {
                "blocks": nn.ModuleList(
                    [
                        Block(
                            args.model_dim,
                            args.num_heads,
                            args.num_kv_heads,
                            args.mlp_mult,
                            args.rope_base,
                            args.qk_gain_init,
                        )
                        for _ in range(args.num_unique_global_blocks)
                    ]
                ),
                "norm": RMSNorm(),
            }
        )
        self.num_global_cycles = args.num_global_cycles

        # Decoder: 2 tiny blocks with causal self-attn + cross-attn to global chunks.
        self.decoder_in = CastedLinear(args.model_dim, args.decoder_dim, bias=False)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=args.decoder_dim,
                    num_heads=args.decoder_heads,
                    memory_dim=args.model_dim,
                    mlp_mult=2,
                    rope_base=args.rope_base,
                    qk_gain_init=args.qk_gain_init,
                )
                for _ in range(args.decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm()
        self.lm_head = CastedLinear(args.decoder_dim, args.vocab_size, bias=False)

        # JEPA head is training-only and intentionally excluded from export.
        self.jepa_predictor = nn.Sequential(
            RMSNorm(),
            CastedLinear(args.model_dim, args.model_dim, bias=False),
            nn.GELU(),
            CastedLinear(args.model_dim, args.model_dim, bias=False),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _encoder_forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tok = self.tok_emb(input_ids)
        tok = tok + self.hash_emb(input_ids)
        tok = F.rms_norm(tok, (tok.size(-1),))
        conv_in = tok.transpose(1, 2)
        conv_in = F.pad(conv_in, (self.args.encoder_kernel_size - 1, 0))
        h = self.encoder_conv(conv_in).transpose(1, 2)
        h = self.encoder_norm(h)

        # Simplified H-Net style boundary predictor from adjacent cosine similarity.
        left = F.normalize(h[:, :-1, :], dim=-1)
        right = F.normalize(h[:, 1:, :], dim=-1)
        cos_adj = (left * right).sum(dim=-1)
        threshold = self.boundary_threshold.to(dtype=cos_adj.dtype)
        boundary_prob = torch.sigmoid((threshold - cos_adj) * self.boundary_temperature)
        boundary_mid = ste_hard_threshold(boundary_prob)
        boundary_start = torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype)
        boundary = torch.cat((boundary_start, boundary_mid), dim=1)
        boundary_hard = (boundary > 0.5).to(h.dtype)
        return h, boundary, boundary_hard

    def _global_forward(self, chunk_latents: Tensor, valid_chunk_mask: Tensor) -> Tensor:
        x = chunk_latents
        x0 = x
        for _ in range(self.num_global_cycles):
            for block in self.global_transformer["blocks"]:
                x = block(x, x0)
                x = x * valid_chunk_mask[:, :, None].to(dtype=x.dtype)
        x = self.global_transformer["norm"](x)
        x = x * valid_chunk_mask[:, :, None].to(dtype=x.dtype)
        return x

    def _decoder_forward(
        self,
        token_latents: Tensor,
        global_chunks: Tensor,
        token_chunk_ids: Tensor,
        valid_chunk_mask: Tensor,
    ) -> Tensor:
        x = self.decoder_in(token_latents)
        chunk_pos = torch.arange(global_chunks.size(1), device=global_chunks.device)[None, None, :]
        allow = chunk_pos <= token_chunk_ids[:, :, None]
        allow = allow & valid_chunk_mask[:, None, :]
        for block in self.decoder_blocks:
            x = block(x, global_chunks, allow)
        x = self.decoder_norm(x)
        logits_proj = self.lm_head(x.reshape(-1, x.size(-1)))
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _jepa_loss(self, global_chunks: Tensor, valid_chunk_mask: Tensor) -> Tensor:
        if global_chunks.size(1) < 2:
            return global_chunks.new_zeros(())
        pred = self.jepa_predictor(global_chunks[:, :-1, :])
        target = global_chunks[:, 1:, :].detach()
        valid_pairs = valid_chunk_mask[:, 1:]
        if not bool(valid_pairs.any()):
            return global_chunks.new_zeros(())
        pred_valid = pred[valid_pairs]
        target_valid = target[valid_pairs]
        mse = F.mse_loss(pred_valid, target_valid, reduction="mean")
        sig = sigreg_loss(pred_valid)
        return mse + self.sigreg_weight * sig

    def export_state_dict(self) -> dict[str, Tensor]:
        # Submission-time export excludes the JEPA predictor head.
        state = self.state_dict()
        return {k: v for k, v in state.items() if not k.startswith("jepa_predictor.")}

    def load_submission_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("jepa_predictor.")]
        missing = [k for k in missing if not k.startswith("jepa_predictor.")]
        if unexpected or missing:
            raise RuntimeError(f"submission state mismatch missing={missing} unexpected={unexpected}")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        token_latents, _boundary_ste, boundary_hard = self._encoder_forward(input_ids)
        chunk_latents, token_chunk_ids, valid_chunk_mask = chunk_reduce(token_latents, boundary_hard)
        global_chunks = self._global_forward(chunk_latents, valid_chunk_mask)
        logits = self._decoder_forward(token_latents, global_chunks, token_chunk_ids, valid_chunk_mask)
        return logits.reshape(input_ids.size(0), input_ids.size(1), self.args.vocab_size)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latents, _boundary_ste, boundary_hard = self._encoder_forward(input_ids)
        chunk_latents, token_chunk_ids, valid_chunk_mask = chunk_reduce(token_latents, boundary_hard)
        global_chunks = self._global_forward(chunk_latents, valid_chunk_mask)
        logits = self._decoder_forward(token_latents, global_chunks, token_chunk_ids, valid_chunk_mask)
        ce = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        if self.training and self.jepa_weight > 0.0:
            return ce + self.jepa_weight * self._jepa_loss(global_chunks, valid_chunk_mask)
        return ce


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_pure_byte_luts(
        args.tokenizer_path, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=byte tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = JEPAByteModel(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token/hash/conv/decoder output heads in Adam
    # - matrix params in attention/MLP projections via Muon
    # - vectors/scalars via Adam
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, p in base_model.named_parameters():
        if name.startswith("jepa_predictor."):
            scalar_params.append(p) if p.ndim < 2 else matrix_params.append(p)
            continue
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    embed_named = (
        "tok_emb.weight",
        "hash_emb.trigram.weight",
        "hash_emb.fourgram.weight",
    )
    embed_params = [dict(base_model.named_parameters())[k] for k in embed_named]
    # Remove embedding params from scalar bucket to keep LR split stable.
    scalar_params = [p for p in scalar_params if p not in embed_params]

    optimizer_embed = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_embed, optimizer_muon, optimizer_scalar, optimizer_head]

    n_params = sum(p.numel() for p in base_model.parameters())
    n_export_params = sum(p.numel() for _, p in base_model.export_state_dict().items())
    log0(f"model_params_total:{n_params} model_params_export:{n_export_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"global_layers_effective:{args.num_unique_global_blocks * args.num_global_cycles}")
    log0(
        f"embed_lr:{args.embed_lr} head_lr:{args.head_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"jepa_weight:{args.jepa_weight} sigreg_weight:{args.sigreg_weight}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int6/8+lzma artifact and validate the round-tripped weights.

    export_state = base_model.export_state_dict()
    if master_process:
        torch.save(export_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_intx(export_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.intx.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.intx.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["intx_payload_bytes"], 1)
        log0(
            f"Serialized model int6/8+lzma: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['intx_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x "
            f"int6_tensors:{quant_stats['num_int6_tensors']} int8_tensors:{quant_stats['num_int8_tensors']})"
        )
        log0(f"Total submission size int6/8+lzma: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.intx.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_submission_state_dict(dequantize_state_dict_intx(quant_state))
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_8_lzma_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_8_lzma_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_8_lzma_ttt_roundtrip val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_int6_8_lzma_ttt_roundtrip_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
