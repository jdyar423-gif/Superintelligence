"""Data + tokenizer preparation for WikiText-2 (raw) subword LM.

- Reads the three parquet splits, concatenates the `text` field exactly (lossless).
- Trains a byte-level BPE tokenizer on TRAIN ONLY (HF `tokenizers`).
- Verifies exact encode->decode round trip on every split (BPB is only valid if lossless).
- Writes token id streams as uint16 .npy plus meta.json (byte counts per split).

Usage: python prep.py --vocab 8192 [--out data/bpe8k]
"""
import argparse, json, os, time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_split(name):
    t = pq.read_table(os.path.join(DATA, f"{name}-00000-of-00001.parquet"))
    return "".join(t.column("text").to_pylist())


def train_bpe(texts, vocab):
    tok = Tokenizer(models.BPE())
    # byte-level, no prefix space => every byte sequence round-trips exactly
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        min_frequency=2,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # feed article-ish chunks (lines) so pre-tokenization is identical to encoding-time
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_stream(tok, text):
    # Encode line-by-line (pre-tokenizer never merges across '\n' anyway) and concatenate.
    # Keeps memory low and is exactly equivalent to encoding the full string because the
    # GPT-2 regex splits on the newline boundary; verified by the round-trip check below.
    lines = text.splitlines(keepends=True)
    encs = tok.encode_batch(lines, add_special_tokens=False)
    ids = [i for e in encs for i in e.ids]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(DATA, f"bpe{args.vocab // 1024}k")
    os.makedirs(out, exist_ok=True)

    splits = {s: load_split(s) for s in ["train", "validation", "test"]}
    t0 = time.time()
    tok = train_bpe(splits["train"].splitlines(keepends=True), args.vocab)
    print(f"trained BPE vocab={tok.get_vocab_size()} in {time.time() - t0:.1f}s")
    tok.save(os.path.join(out, "tokenizer.json"))

    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": tok.token_to_id("<|eos|>")}
    for s, text in splits.items():
        ids = encode_stream(tok, text)
        dec = tok.decode(ids)
        assert dec == text, f"round trip failed on {s}"
        arr = np.array(ids, dtype=np.uint16)
        np.save(os.path.join(out, f"{s}.npy"), arr)
        nbytes = len(text.encode("utf-8"))
        meta[s] = {"tokens": len(ids), "bytes": nbytes, "bytes_per_token": nbytes / len(ids)}
        print(f"{s:10s} tokens={len(ids):9d} bytes={nbytes:9d} bytes/token={nbytes / len(ids):.3f}  (lossless OK)")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
