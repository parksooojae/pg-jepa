"""
Convert sp1024 (SentencePiece BPE) dataset to byte260 (raw bytes) format
and upload to HuggingFace Hub.

Reads sp1024 .bin files, decodes tokens back to text via SentencePiece,
then encodes as raw bytes (0-255) and uploads to HuggingFace.

Usage:
    python convert_sp1024_to_byte260.py --repo-id YOUR_USERNAME/fineweb10B_byte260
    
    # Or with explicit token
    HF_TOKEN=hf_xxx python convert_sp1024_to_byte260.py --repo-id YOUR_USERNAME/fineweb10B_byte260
"""

import argparse
import glob
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
SP_MODEL = ROOT / "tokenizers" / "fineweb_1024_bpe.model"
SP1024_DIR = ROOT / "datasets" / "fineweb10B_sp1024"

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
    vocab_size = sp.get_piece_size()
    token_list = tokens.tolist()
    
    # Debug: show token range
    print(f"  Token range: min={min(token_list)}, max={max(token_list)}, vocab_size={vocab_size}")
    
    # Decode in chunks to avoid memory issues
    chunk_size = 100_000
    all_bytes = []
    
    for i in range(0, len(token_list), chunk_size):
        chunk = token_list[i:i + chunk_size]
        # Filter out tokens that are out of vocab range (special tokens like BOS/EOS/PAD)
        chunk = [t for t in chunk if t < vocab_size]
        if chunk:
            text = sp.decode(chunk)
            # Convert text to bytes (UTF-8)
            text_bytes = text.encode('utf-8', errors='replace')
            all_bytes.extend(text_bytes)
    
    # Convert to uint16 array (byte values 0-255)
    return np.array(all_bytes, dtype=np.uint16)


def convert_and_upload(
    sp: spm.SentencePieceProcessor,
    src_path: str,
    api: HfApi,
    repo_id: str,
) -> None:
    """Convert a single sp1024 file to byte260 format and upload."""
    filename = Path(src_path).name
    print(f"Converting {src_path}")
    
    tokens = load_sp1024_tokens(src_path)
    print(f"  Loaded {len(tokens):,} sp1024 tokens")
    
    byte_tokens = decode_and_convert(sp, tokens)
    print(f"  Converted to {len(byte_tokens):,} bytes")
    
    # Upload directly from memory
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        byte_tokens.tofile(tmp.name)
        api.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo=f"datasets/fineweb10B_byte260/{filename}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        os.unlink(tmp.name)
    
    print(f"  Uploaded to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert sp1024 to byte260 and upload to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g., username/fineweb10B_byte260)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()
    
    # Check SentencePiece model exists
    if not SP_MODEL.exists():
        print(f"ERROR: SentencePiece model not found at {SP_MODEL}")
        print("Run: python cached_challenge_fineweb.py --variant sp1024 --train-shards 0")
        return
    
    # Check source directory exists
    if not SP1024_DIR.exists():
        print(f"ERROR: Source directory not found at {SP1024_DIR}")
        return
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Create repo if it doesn't exist
    print(f"Creating/verifying repo: {args.repo_id}")
    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    
    # Load SentencePiece model
    print(f"Loading SentencePiece model from {SP_MODEL}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL))
    print(f"  Vocab size: {sp.get_piece_size()}")
    
    # Find all sp1024 .bin files
    train_files = sorted(glob.glob(str(SP1024_DIR / "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(str(SP1024_DIR / "fineweb_val_*.bin")))
    
    print(f"Found {len(train_files)} train files, {len(val_files)} val files")
    
    # Convert and upload validation files
    for src in val_files:
        convert_and_upload(sp, src, api, args.repo_id)
    
    # Convert and upload training files
    for src in train_files:
        convert_and_upload(sp, src, api, args.repo_id)
    
    print(f"\nDone! Byte260 dataset uploaded to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
