import sys
import os
import time
import zlib
import lzma
import brotli
import zstandard as zstd
from datasets import load_dataset
from tqdm import tqdm

# 親ディレクトリのモジュールを読み込めるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semantic_compressor.strategies import pack_tokens, ChimeraTokenStrategy
from tokenizers import Tokenizer

def benchmark_algorithm(name, stats, encode_func, data):
    t0 = time.perf_counter()
    try:
        compressed = encode_func(data)
        size = len(compressed)
    except Exception as e:
        size = len(data) # Fallback
    t1 = time.perf_counter()

    stats[name]["bytes"] += size
    stats[name]["time_ms"] += (t1 - t0) * 1000

def main():
    print("Loading Dolly-15k-ja dataset...")
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")

    tokenizer_path = "./tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    bits_per_id = 18
    chimera = ChimeraTokenStrategy()

    # 圧縮器の準備 (最高・高圧縮設定)
    zstd_comp = zstd.ZstdCompressor(level=19) # Zstd高圧縮

    stats = {
        "Raw UTF-8": {"bytes": 0, "time_ms": 0},
        "Zlib (Raw)": {"bytes": 0, "time_ms": 0},
        "Brotli (Raw)": {"bytes": 0, "time_ms": 0},
        "LZMA (Raw)": {"bytes": 0, "time_ms": 0},
        "Zstd (Raw)": {"bytes": 0, "time_ms": 0},

        "Raw Tokens (18bit)": {"bytes": 0, "time_ms": 0},
        "Zlib (Tokens)": {"bytes": 0, "time_ms": 0},
        "Brotli (Tokens)": {"bytes": 0, "time_ms": 0},
        "LZMA (Tokens)": {"bytes": 0, "time_ms": 0},
        "Zstd (Tokens)": {"bytes": 0, "time_ms": 0},
        "Chimera (Tokens)": {"bytes": 0, "time_ms": 0},
    }

    # 処理時間がかかりすぎるため、先頭2000件で限界突破テストを行う
    limit = 2000
    print(f"Starting ULTIMATE benchmark on {limit} rows (Max Compression Settings)...")

    for i, row in enumerate(tqdm(dataset, total=limit, desc="Compressing")):
        if i >= limit:
            break

        text = f"{row['instruction']}\n{row['input']}\n{row['output']}".strip()
        if not text:
            continue

        raw_bytes = text.encode('utf-8')
        if len(raw_bytes) == 0:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False).ids

        # --- Raw Text Algorithms ---
        stats["Raw UTF-8"]["bytes"] += len(raw_bytes)

        benchmark_algorithm("Zlib (Raw)", stats, lambda d: zlib.compress(d, level=9), raw_bytes)
        benchmark_algorithm("Brotli (Raw)", stats, lambda d: brotli.compress(d, quality=11), raw_bytes)
        benchmark_algorithm("LZMA (Raw)", stats, lambda d: lzma.compress(d, preset=9), raw_bytes)
        benchmark_algorithm("Zstd (Raw)", stats, lambda d: zstd_comp.compress(d), raw_bytes)

        # --- Semantic (Token) Algorithms ---
        packed_tokens = pack_tokens(token_ids, bits_per_id)
        stats["Raw Tokens (18bit)"]["bytes"] += len(packed_tokens)

        benchmark_algorithm("Zlib (Tokens)", stats, lambda d: zlib.compress(d, level=9), packed_tokens)
        benchmark_algorithm("Brotli (Tokens)", stats, lambda d: brotli.compress(d, quality=11), packed_tokens)
        benchmark_algorithm("LZMA (Tokens)", stats, lambda d: lzma.compress(d, preset=9), packed_tokens)
        benchmark_algorithm("Zstd (Tokens)", stats, lambda d: zstd_comp.compress(d), packed_tokens)

        t0 = time.perf_counter()
        chimera_bytes = chimera.encode(text, token_ids, bits_per_id)
        t1 = time.perf_counter()
        stats["Chimera (Tokens)"]["bytes"] += len(chimera_bytes)
        stats["Chimera (Tokens)"]["time_ms"] += (t1 - t0) * 1000

    print("\n" + "="*70)
    print(" 🏆 ULTIMATE BENCHMARK RESULTS (Top Compressors vs Semantic)")
    print("="*70)

    raw_total = stats["Raw UTF-8"]["bytes"]
    print(f"Total Original Size : {raw_total / 1024 / 1024:.2f} MB")
    print("-" * 70)
    print(f"{'Algorithm':<25} | {'Size (MB)':<10} | {'Ratio (%)':<10} | {'Time (Total s)':<15}")
    print("-" * 70)

    # Sort by compression ratio (lowest size first)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["bytes"])

    for name, data in sorted_stats:
        size_mb = data["bytes"] / 1024 / 1024
        ratio = (data["bytes"] / raw_total) * 100
        time_s = data["time_ms"] / 1000

        # Highlight token-based semantic methods
        marker = "🟢" if "(Tokens)" in name else "⚪"
        print(f"{marker} {name:<23} | {size_mb:>8.2f} | {ratio:>8.2f} % | {time_s:>12.2f} s")

    print("-" * 70)
    print("🟢 = Semantic / Token-based approach")
    print("⚪ = Standard Text Compression")

if __name__ == '__main__':
    main()
