import time
import zlib
from datasets import load_dataset
from tqdm import tqdm
from semantic_compressor import SemanticCompressor

def main():
    print("Loading Dolly-15k-ja dataset...")
    # Load the Japanese Dolly 15k dataset
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    compressor = SemanticCompressor("tokenizer.json")
    
    total_raw_bytes = 0
    total_zlib_bytes = 0
    total_semantic_bytes = 0
    
    count = 0
    semantic_wins = 0
    zlib_wins = 0
    ties = 0
    
    print(f"Starting benchmark on {len(dataset)} rows...")
    t_start = time.perf_counter()
    
    for row in tqdm(dataset, desc="Compressing", unit="row"):
        # Create a single string for the row
        text = f"{row['instruction']}\n{row['input']}\n{row['output']}".strip()
        if not text:
            continue
            
        raw_bytes = text.encode('utf-8')
        raw_size = len(raw_bytes)
        if raw_size == 0:
            continue
            
        # 1. Zlib on Raw UTF-8
        zlib_compressed = zlib.compress(raw_bytes, level=9)
        zlib_size = len(zlib_compressed)
        
        # 2. Semantic Compressor
        try:
            semantic_compressed = compressor.encode(text=text)
            semantic_size = len(semantic_compressed)
        except Exception as e:
            # Fallback if tokenizer fails or decoding fails
            semantic_size = raw_size
            
        total_raw_bytes += raw_size
        total_zlib_bytes += zlib_size
        total_semantic_bytes += semantic_size
        count += 1
        
        # Compare per row
        if semantic_size < zlib_size:
            semantic_wins += 1
        elif zlib_size < semantic_size:
            zlib_wins += 1
        else:
            ties += 1

    t_end = time.perf_counter()
    
    # ---------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" Benchmark Results: Semantic Compressor vs Zlib")
    print("="*50)
    print(f"Total Rows Processed : {count:,}")
    print(f"Total Processing Time: {t_end - t_start:.2f} seconds")
    
    print("\n[Total Size]")
    print(f"Original Text (UTF-8): {total_raw_bytes / 1024 / 1024:.2f} MB ({total_raw_bytes:,} bytes)")
    print(f"Zlib (Level 9)       : {total_zlib_bytes / 1024 / 1024:.2f} MB ({total_zlib_bytes:,} bytes)")
    print(f"Semantic Compressor  : {total_semantic_bytes / 1024 / 1024:.2f} MB ({total_semantic_bytes:,} bytes)")
    
    print("\n[Compression Ratio (Lower is better)]")
    zlib_ratio = (total_zlib_bytes / total_raw_bytes) * 100
    sem_ratio = (total_semantic_bytes / total_raw_bytes) * 100
    print(f"Zlib Ratio           : {zlib_ratio:.2f} %")
    print(f"Semantic Ratio       : {sem_ratio:.2f} %")
    
    print("\n[Space Saved]")
    zlib_saved = total_raw_bytes - total_zlib_bytes
    sem_saved = total_raw_bytes - total_semantic_bytes
    print(f"Zlib Saved           : {zlib_saved / 1024 / 1024:.2f} MB")
    print(f"Semantic Saved       : {sem_saved / 1024 / 1024:.2f} MB")
    
    print("\n[Win/Loss (Per Row)]")
    print(f"Semantic Wins        : {semantic_wins:,} rows")
    print(f"Zlib Wins            : {zlib_wins:,} rows")
    print(f"Ties                 : {ties:,} rows")
    
    print("\n[Conclusion]")
    if total_semantic_bytes < total_zlib_bytes:
        diff = total_zlib_bytes - total_semantic_bytes
        print(f"★ Semantic Compressor is overall MORE EFFICIENT by {diff / 1024:.2f} KB.")
    else:
        diff = total_semantic_bytes - total_zlib_bytes
        print(f"★ Zlib is overall MORE EFFICIENT by {diff / 1024:.2f} KB.")

if __name__ == '__main__':
    main()
