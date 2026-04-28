from semantic_compressor import SemanticCompressor

def main():
    # ライブラリの初期化 (Tokenizerのパスを指定)
    compressor = SemanticCompressor("tokenizer.json")
    
    test_text = "こんにちは！これはセマンティック圧縮ライブラリのテストです。0000000000"
    print("元のテキスト:", test_text)
    
    # パターン1: 変数で受け取る
    print("\n--- パターン1: オンメモリ処理 ---")
    compressed_bytes = compressor.encode(text=test_text)
    print(f"圧縮後のバイナリサイズ: {len(compressed_bytes)} bytes")
    
    decoded_text = compressor.decode(payload=compressed_bytes)
    print("復元成功！" if test_text == decoded_text else "復元失敗...")
    
    # パターン2: ファイルに直接入出力する
    print("\n--- パターン2: ファイルI/O処理 ---")
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
        
    compressor.encode(input_filepath="input.txt", output_filepath="output.bin")
    print("output.bin を出力しました。")
    
    compressor.decode(input_filepath="output.bin", output_filepath="restored.txt")
    print("restored.txt に復元しました。")

if __name__ == "__main__":
    main()
