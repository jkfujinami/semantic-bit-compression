from tokenizers import Tokenizer
import json
import os
import math

def main():
    # tokenizer.json のパス
    tokenizer_path = "tokenizer.json"

    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found.")
        return

    # トークナイザーの読み込み
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # テスト用のテキスト
    test_text = "お金が欲しいわ。10000万円の"

    # トークン化の実行 (特殊トークンを追加しない)
    encoding = tokenizer.encode(test_text, add_special_tokens=False)

    # ビット数の計算
    # 1. Token IDs のビット数
    bits_per_id = math.ceil(math.log2(vocab_size))
    total_token_bits = len(encoding.ids) * bits_per_id

    # 2. 生テキストのビット数 (UTF-8)
    raw_bytes = test_text.encode('utf-8')
    total_raw_bits = len(raw_bytes) * 8

    # 結果の表示
    print(f"Input Text: {test_text}")
    print(f"Vocab Size: {vocab_size} (Minimum bits per ID: {bits_per_id})")
    print("-" * 50)

    # 1. 生テキストのビット表示
    print("Raw Text (UTF-8) Bits:")
    raw_bits_str = "".join(format(b, '08b') for b in raw_bytes)
    # 8ビットごとにスペースを入れて見やすく表示
    formatted_raw_bits = " ".join(raw_bits_str[i:i+8] for i in range(0, len(raw_bits_str), 8))
    print(f"  {formatted_raw_bits}")
    print(f"  Total: {total_raw_bits} bits")
    print("-" * 50)

    # 2. Token IDs のビット表示
    print(f"Token IDs Binary ({bits_per_id}bit fixed):")
    token_bits_str = "".join(format(idx, f'0{bits_per_id}b') for idx in encoding.ids)
    # bits_per_idごとにスペースを入れて表示
    formatted_token_bits = " ".join(token_bits_str[i:i+bits_per_id] for i in range(0, len(token_bits_str), bits_per_id))
    print(f"  {formatted_token_bits}")
    print(f"  Total: {total_token_bits} bits")
    print("-" * 50)

    # 3. 圧縮率
    if total_raw_bits > 0:
        ratio = total_token_bits / total_raw_bits
        print(f"Compression Ratio (Token/Raw): {ratio:.2%} (Saved {total_raw_bits - total_token_bits} bits)")

    print("-" * 50)
    decording = tokenizer.decode(encoding.ids)
    print(f"Decoded: {decording}")

if __name__ == "__main__":
    main()
