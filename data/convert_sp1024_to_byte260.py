"""
Convert sp1024 (SentencePiece BPE) dataset to byte260 (raw bytes) format.

Reads sp1024 .bin files, decodes tokens back to text via SentencePiece,
then encodes as raw bytes (0-255) with special tokens (256-259).

Usage:
    python convert_sp1024_to_byte260.py
"""

import glob
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
SP_MODEL = ROOT / "tokenizers" / "fineweb_1024_bpe.model"
SP1024_DIR = ROOT / "datasets" / "fineweb10B_sp1024"
BYTE260_DIR = ROOT / "datasets" / "fineweb10B_byte260"

# byte260 special tokens (matching train_jepa.py expectations)
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259


def load_sp1024_tokens(path: str) -> np.ndarray:
    """Load sp1024 .bin file as uint16 tokens."""
    return np.fromfile(path, dtype=np.uint16)


def decode_and_convert(sp: spm.SentencePieceProcessor, tokens: np.ndarray) -> np.ndarray:
    """Decode sp1024 tokens to text, then encode as bytes."""
    # Decode tokens to text (skip BOS/EOS tokens from sp1024)
    # sp1024 uses BOS=1, EOS=2
    token_list = tokens.tolist()
    
    # Decode in chunks to avoid memory issues
    chunk_size = 100_000
    all_bytes = []
    
    for i in range(0, len(token_list), chunk_size):
        chunk = token_list[i:i + chunk_size]
        # Filter out BOS/EOS tokens before decoding
        chunk = [t for t in chunk if t not in (1, 2)]
        if chunk:
            text = sp.decode(chunk)
            # Convert text to bytes (UTF-8)
            text_bytes = text.encode('utf-8', errors='replace')
            all_bytes.extend(text_bytes)
    
    # Convert to uint16 array (byte values 0-255)
    return np.array(all_bytes, dtype=np.uint16)


def convert_file(sp: spm.SentencePieceProcessor, src_path: str, dst_path: str) -> None:
    """Convert a single sp1024 file to byte260 format."""
    print(f"Converting {src_path} -> {dst_path}")
    
    tokens = load_sp1024_tokens(src_path)
    print(f"  Loaded {len(tokens):,} sp1024 tokens")
    
    byte_tokens = decode_and_convert(sp, tokens)
    print(f"  Converted to {len(byte_tokens):,} bytes")
    
    # Save as uint16 (matching sp1024 format for compatibility)
    byte_tokens.tofile(dst_path)
    print(f"  Saved to {dst_path}")


def main():
    # Check SentencePiece model exists
    if not SP_MODEL.exists():
        print(f"ERROR: SentencePiece model not found at {SP_MODEL}")
        print("Run: python cached_challenge_fineweb.py --variant sp1024 --train-shards 0")
        return
    
    # Check source directory exists
    if not SP1024_DIR.exists():
        print(f"ERROR: Source directory not found at {SP1024_DIR}")
        return
    
    # Create output directory
    BYTE260_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SentencePiece model
    print(f"Loading SentencePiece model from {SP_MODEL}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL))
    print(f"  Vocab size: {sp.get_piece_size()}")
    
    # Find all sp1024 .bin files
    train_files = sorted(glob.glob(str(SP1024_DIR / "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(str(SP1024_DIR / "fineweb_val_*.bin")))
    
    print(f"Found {len(train_files)} train files, {len(val_files)} val files")
    
    # Convert validation files
    for src in val_files:
        dst = str(BYTE260_DIR / Path(src).name)
        convert_file(sp, src, dst)
    
    # Convert training files
    for src in train_files:
        dst = str(BYTE260_DIR / Path(src).name)
        convert_file(sp, src, dst)
    
    print("\nDone! Byte260 dataset created at:", BYTE260_DIR)


if __name__ == "__main__":
    main()
