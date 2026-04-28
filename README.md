# semantic-bit-compression
An extreme text compression format utilizing LLM tokenizer and variable-length coding.

```
======================================================================
 🏆 ULTIMATE BENCHMARK RESULTS (Top Compressors vs Semantic)
======================================================================
Total Original Size : 1.93 MB
----------------------------------------------------------------------
Algorithm                 | Size (MB)  | Ratio (%)  | Time (Total s)
----------------------------------------------------------------------
🟢 BWT+RLE-Zero (Tokens)   |     0.76 |    39.25 % |        74.35 s
🟢 Pure BWT+MTF (Tokens)   |     0.76 |    39.66 % |        74.86 s
🟢 BWT+RLEZ NoZlib (Tokens) |     0.77 |    40.20 % |        74.85 s
🟢 Chimera (Tokens)        |     0.79 |    41.19 % |         7.60 s
⚪ Brotli (Raw)            |     0.84 |    43.53 % |         2.06 s
🟢 Zstd (Tokens)           |     0.84 |    43.80 % |         0.07 s
🟢 Brotli (Tokens)         |     0.85 |    43.93 % |         2.12 s
🟢 Zlib (Tokens)           |     0.86 |    44.41 % |         0.06 s
🟢 BWT+MTF (Tokens)        |     0.89 |    46.28 % |         3.18 s
⚪ Raw Tokens (18bit)      |     0.90 |    46.70 % |         0.00 s
⚪ Zlib (Raw)              |     0.94 |    48.64 % |         0.08 s
🟢 LZMA (Tokens)           |     0.95 |    49.45 % |         1.52 s
⚪ Zstd (Raw)              |     0.96 |    50.04 % |         0.25 s
🟢 BWT+Chimera (Tokens)    |     1.01 |    52.47 % |        83.99 s
⚪ LZMA (Raw)              |     1.04 |    53.96 % |         1.83 s
🟢 Chimera-MTF (Tokens)    |     1.04 |    54.01 % |        82.34 s
⚪ Raw UTF-8               |     1.93 |   100.00 % |         0.00 s
----------------------------------------------------------------------
🟢 = Semantic / Token-based approach
⚪ = Standard Text Compression
```