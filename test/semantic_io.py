import time
import os
import zlib
import struct
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass
from tokenizers import Tokenizer

# ==========================================
# 1. Models & Bitwise Utilities
# ==========================================
class BitWriter:
    def __init__(self):
        self.accumulator = 0
        self.bit_count = 0
        self.byte_array = bytearray()

    def write(self, value: int, num_bits: int):
        self.accumulator = (self.accumulator << num_bits) | (value & ((1 << num_bits) - 1))
        self.bit_count += num_bits
        while self.bit_count >= 8:
            self.bit_count -= 8
            byte_val = (self.accumulator >> self.bit_count) & 0xFF
            self.byte_array.append(byte_val)

    def get_bytes(self) -> bytes:
        if self.bit_count > 0:
            byte_val = (self.accumulator << (8 - self.bit_count)) & 0xFF
            self.byte_array.append(byte_val)
        return bytes(self.byte_array)

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_idx = 0
        self.accumulator = 0
        self.bit_count = 0

    def read(self, num_bits: int) -> int:
        while self.bit_count < num_bits:
            if self.byte_idx < len(self.data):
                self.accumulator = (self.accumulator << 8) | self.data[self.byte_idx]
                self.byte_idx += 1
                self.bit_count += 8
            else:
                self.accumulator <<= (num_bits - self.bit_count)
                self.bit_count = num_bits
                break

        self.bit_count -= num_bits
        val = (self.accumulator >> self.bit_count) & ((1 << num_bits) - 1)
        return val

@dataclass
class EncodeResult:
    strategy_id: int
    strategy_name: str
    payload: bytes
    encode_time_ms: float
    decode_time_ms: float

# ==========================================
# 2. Interfaces
# ==========================================
class ICompressionStrategy(ABC):
    @property
    @abstractmethod
    def strategy_id(self) -> int: pass

    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes: pass

    @abstractmethod
    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str: pass

# ==========================================
# 3. Strategies
# ==========================================
class RawUtf8Strategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 0
    @property
    def name(self) -> str: return "Raw UTF-8 (Fallback)"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        return text.encode('utf-8')

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        return payload.decode('utf-8')

class ZlibRawUtf8Strategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 1
    @property
    def name(self) -> str: return "Zlib (Deflate) on Raw UTF-8"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        return zlib.compress(text.encode('utf-8'), level=9)

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        return zlib.decompress(payload).decode('utf-8')

def pack_tokens(ids: List[int], bits_per_id: int) -> bytes:
    writer = BitWriter()
    for token_id in ids:
        writer.write(token_id, bits_per_id)
    return writer.get_bytes()

def unpack_tokens(data: bytes, bits_per_id: int, count: int) -> List[int]:
    reader = BitReader(data)
    ids = []
    for _ in range(count):
        ids.append(reader.read(bits_per_id))
    return ids

class RawTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 2
    @property
    def name(self) -> str: return "Raw Tokens (18-bit native)"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        return pack_tokens(token_ids, bits_per_id)

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        ids = unpack_tokens(payload, bits_per_id, token_count)
        return tokenizer.decode(ids)

class ZlibTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 3
    @property
    def name(self) -> str: return "Zlib (Deflate) on Tokens"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        return zlib.compress(pack_tokens(token_ids, bits_per_id), level=9)

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        unzipped = zlib.decompress(payload)
        ids = unpack_tokens(unzipped, bits_per_id, token_count)
        return tokenizer.decode(ids)

class ChimeraTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 4
    @property
    def name(self) -> str: return "Chimera (Escape-based RLE + LZ77)"

    def _get_params(self, bits_per_id: int):
        return {
            "ESCAPE_ID": (1 << bits_per_id) - 1,
            "cmd_bits": 2,
            "rle_count_bits": 8,
            "lz_dist_bits": 12,
            "lz_len_bits": 8
        }

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        writer = BitWriter()
        p = self._get_params(bits_per_id)

        cost_rle = bits_per_id + p["cmd_bits"] + bits_per_id + p["rle_count_bits"]
        cost_lz77 = bits_per_id + p["cmd_bits"] + p["lz_dist_bits"] + p["lz_len_bits"]

        i = 0
        window_size = (1 << p["lz_dist_bits"]) - 1
        max_lz_len = (1 << p["lz_len_bits"]) - 1
        max_rle_len = (1 << p["rle_count_bits"]) - 1

        while i < len(token_ids):
            # 1. RLE
            rle_len = 1
            while i + rle_len < len(token_ids) and token_ids[i + rle_len] == token_ids[i] and rle_len < max_rle_len:
                rle_len += 1
            rle_savings = (rle_len * bits_per_id) - cost_rle

            # 2. LZ77
            lz_len = 0
            lz_dist = 0
            start_w = max(0, i - window_size)
            for j in range(start_w, i):
                l = 0
                while i + l < len(token_ids) and token_ids[j + l] == token_ids[i + l] and l < max_lz_len:
                    l += 1
                if l > lz_len:
                    lz_len = l
                    lz_dist = i - j
            lz_savings = (lz_len * bits_per_id) - cost_lz77

            # 3. Greedy Choice
            if max(rle_savings, lz_savings) > 0:
                if rle_savings >= lz_savings:
                    writer.write(p["ESCAPE_ID"], bits_per_id)
                    writer.write(1, p["cmd_bits"])
                    writer.write(token_ids[i], bits_per_id)
                    writer.write(rle_len, p["rle_count_bits"])
                    i += rle_len
                else:
                    writer.write(p["ESCAPE_ID"], bits_per_id)
                    writer.write(2, p["cmd_bits"])
                    writer.write(lz_dist, p["lz_dist_bits"])
                    writer.write(lz_len, p["lz_len_bits"])
                    i += lz_len
            else:
                if token_ids[i] == p["ESCAPE_ID"]:
                    writer.write(p["ESCAPE_ID"], bits_per_id)
                    writer.write(0, p["cmd_bits"])
                else:
                    writer.write(token_ids[i], bits_per_id)
                i += 1

        return writer.get_bytes()

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        reader = BitReader(payload)
        p = self._get_params(bits_per_id)
        ids = []

        while len(ids) < token_count:
            val = reader.read(bits_per_id)
            if val == p["ESCAPE_ID"]:
                cmd = reader.read(p["cmd_bits"])
                if cmd == 0:
                    ids.append(p["ESCAPE_ID"])
                elif cmd == 1:
                    token = reader.read(bits_per_id)
                    rle_count = reader.read(p["rle_count_bits"])
                    ids.extend([token] * rle_count)
                elif cmd == 2:
                    dist = reader.read(p["lz_dist_bits"])
                    length = reader.read(p["lz_len_bits"])
                    start_idx = len(ids) - dist
                    for k in range(length):
                        ids.append(ids[start_idx + k])
            else:
                ids.append(val)

        return tokenizer.decode(ids)

# ==========================================
# 4. Core System
# ==========================================
class SemanticCompressorSystem:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bits_per_id = 18
        self.strategies = [
            RawUtf8Strategy(),
            ZlibRawUtf8Strategy(),
            RawTokenStrategy(),
            ZlibTokenStrategy(),
            ChimeraTokenStrategy()
        ]
        self.strat_map = {s.strategy_id: s for s in self.strategies}

    def encode_file(self, input_filepath: str, output_binpath: str):
        with open(input_filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()

        token_ids = self.tokenizer.encode(original_text, add_special_tokens=False).ids
        token_count = len(token_ids)

        results = []

        for strat in self.strategies:
            t0 = time.perf_counter()
            payload = strat.encode(original_text, token_ids, self.bits_per_id)
            t1 = time.perf_counter()

            t2 = time.perf_counter()
            decoded_text = strat.decode(payload, self.tokenizer, self.bits_per_id, token_count)
            t3 = time.perf_counter()

            encode_ms = (t1 - t0) * 1000
            decode_ms = (t3 - t2) * 1000

            # Lossless verification: Tokenization decode often removes spaces depending on tokenizer.
            # If the decoded text doesn't match original EXACTLY, we must penalize it or skip it.
            if original_text != decoded_text and strat.strategy_id != 0:
                print(f"[Warning] {strat.name} is Lossy (tokenizer decode mismatch). Skipping.")
                continue

            results.append(EncodeResult(
                strategy_id=strat.strategy_id,
                strategy_name=strat.name,
                payload=payload,
                encode_time_ms=encode_ms,
                decode_time_ms=decode_ms
            ))

        raw_res = next(r for r in results if r.strategy_id == 0)

        # Best Selection Algorithm (smallest payload)
        best_res = raw_res
        for r in results:
            if len(r.payload) < len(best_res.payload):
                best_res = r

        # Write to .bin
        # Header: [1 byte ID] + [4 bytes uint32 Token Count]
        header = struct.pack('<BI', best_res.strategy_id, token_count)
        final_binary = header + best_res.payload

        with open(output_binpath, 'wb') as f:
            f.write(final_binary)

        return original_text, token_count, results, best_res, final_binary

    def decode_file(self, bin_filepath: str, output_filepath: str):
        with open(bin_filepath, 'rb') as f:
            data = f.read()

        strategy_id, token_count = struct.unpack('<BI', data[:5])
        payload = data[5:]

        strat = self.strat_map[strategy_id]

        t0 = time.perf_counter()
        decoded_text = strat.decode(payload, self.tokenizer, self.bits_per_id, token_count)
        t1 = time.perf_counter()

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(decoded_text)

        return decoded_text, strat.name, (t1 - t0) * 1000

if __name__ == "__main__":
    system = SemanticCompressorSystem("tokenizer.json")

    # Generate a test text file (chat log mix)
    test_txt = "test_input.txt"
    with open(test_txt, "w", encoding="utf-8") as f:
        f.write("08:32 焼肉 今回の制作はgoの勉強も兼ねているし、車輪の再発明にも意味はあります💢\n")
        f.write("00000000000000000000\n")
        f.write("""---
title: 【初学者向け】「どう並べて、どう取り出すか」で変わる - データ構造の基礎
tags: 初心者 エンジニア 初学者向け 未経験エンジニア 新人プログラマ応援
author: masa20057
slide: false
---

株式会社PRUMのmasaです。

今日は「データをどう並べ、どう取り出すか」というルール（データ構造）について、初学者向けにまとめました。

プログラムは「データの集まり」をどう扱うかで、その動き方が大きく変わります。

この記事では、データの扱い方の基本パターンを知ることで、「どのようにデータを整理し、どう処理すべきか」を判断するための土台を身につけることを目的としています。

## 処理の順番で使い分ける　「スタック」と「キュー」の基本

プログラミングをしていると、複数のデータを一時的に溜めておき、後で順番に取り出して処理したい場面によく遭遇します。

このとき、「どういう順番でデータを取り出すか」というルールを定義するのが、**スタック** と **キュー** というデータ構造です。

### 結論：出口のルールで「できる処理」が変わる

「最後に入れたものを先に出すか（スタック）」、それとも「最初に入れたものから出すか（キュー）」という違いによって、実現できる処理の形が決まります。

この違いを理解することで、処理の流れが整理され、特に順序が重要なロジックにおいて、意図しない挙動（バグ）を防ぎやすくなります。

### 基礎知識：LIFOとFIFO

言葉だけ聞くと難しそうですが、仕組みはシンプルです。

* **スタック（Stack）** ： **LIFO**（Last-In, First-Out / 後入れ先出し）
  → 最後に保存したデータが、最初に取り出される

* **キュー（Queue）** ： **FIFO**（First-In, First-Out / 先入れ先出し）
  → 最初に保存したデータが、順番に取り出される

### 具体例：机の上の書類とスーパーのレジ待ち

**スタックは「机の上に積み上げた書類」です。**
最後に置いた書類が一番上に来るため、次に使うときは一番最初に取り出されます。

ブラウザの「戻る」ボタンもこの考え方に近く、最後に見たページに戻ります。
※実際には「戻る用」と「進む用」で複数のスタックを組み合わせて実現されています。

**キューは「スーパーのレジ待ち」です。**
先に並んだ人から順番に案内されます。

コンピュータでは、プリンターの印刷待ちなどで使われています。

## 複数データを効率よく扱う　「配列」と「リスト」の使い分け

次に、複数のデータをメモリ上にどう並べるかという話です。
よく使われるのが **配列（Array）** と **リスト（連結リスト）** です。

### 結論：やりたい操作で「収納方法」を選ぶ

データ構造には、それぞれ得意・不得意があります。
「どんな操作を一番よく行うか」で選ぶことが重要です。

### 基礎知識：メモリ上の並び方の違い

* **配列（Array）**
  メモリ上に「連続した領域」として並んでいます
  （言語によって内部実装は異なりますが、基本的な考え方は同じです）

* **リスト（連結リスト / Linked List）**
  データがバラバラな場所にあり、それぞれが「次のデータの場所」を持っています
  （この「次を指す仕組み」をポインタと呼びます）

### 具体例：マンションの郵便受けと連絡網

**配列は「番号が決まっているマンションの郵便受け」です。**

101号室、102号室…と並んでいるので、
「105号室」と言われれば一瞬で見つけられます。

* **メリット**：特定の位置のデータを高速に取得できる
* **デメリット**：途中に要素を入れるとき、後ろのデータをずらす必要がありコストが高い

**リストは「電話の連絡網」です。**

「Aさん→Bさん→Cさん」と順番に繋がっています。

* **メリット**：途中にデータを追加・削除しやすい（つなぎ替えるだけ）
* **デメリット**：目的のデータにたどり着くには順番に追う必要がある

### どちらを使うべきかの判断基準

迷ったときは、 **どの操作を一番よく行うか** で考えます。

> 特定の位置のデータをすぐ取り出したい

→ 配列

> 途中への追加・削除が多い

→ リスト

ただし実務では、CPUキャッシュ効率などの理由から、配列（動的配列）が使われるケースが多いです。

## さいごに
データ構造は一見地味ですが、「データをどう扱うか」というプログラムの根本的な考え方を支える重要な土台です。

いきなりすべてを理解する必要はありません。
まずは「このデータはどんなルールで扱うと自然か？」という視点を持つだけで、コードの見え方が変わってきます。

少しずつ経験を積みながら、自分なりに「扱いやすい形」を選べるようになっていきましょう。

---
PRUMのエンジニアの95%以上は未経験からの採用です。
よければコーポレートサイトにも遊びに来てください。
▶ [コーポレートサイト](https://prum.jp/recruit?utm_source=qiita&utm_medium=article&utm_campaign=masa_20260428&utm_content=recruit)

エンジニアの方に役立つ記事をまとめたサイトも運営しています。もしご興味あれば覗いてみてくださいね。
▶ [エンジニアに役立つ記事サイト](https://prum.jp/01engineer/?utm_source=qiita&utm_medium=article&utm_campaign=masa_20260428&utm_content=om)
""")


    print("--- [ENCODE PHASE] ---")
    orig_text, token_count, all_res, best_res, final_bin = system.encode_file(test_txt, "output.bin")

    raw_size = len(orig_text.encode('utf-8'))
    print(f"Original Text: {raw_size} bytes")
    print(f"Token Count: {token_count}")
    print("\n[Strategy Evaluations]")
    for r in all_res:
        indicator = "★ BEST" if r.strategy_id == best_res.strategy_id else "      "
        ratio = (len(r.payload) / raw_size) * 100
        print(f"{indicator} | {len(r.payload):>6} bytes ({ratio:>5.1f}%) | Enc: {r.encode_time_ms:>6.3f} ms | Dec: {r.decode_time_ms:>6.3f} ms | {r.strategy_name}")

    print(f"\n[Writing File]")
    print(f"Algorithm selected: {best_res.strategy_name} (ID: 0x{best_res.strategy_id:02X})")
    print(f"Final .bin written: output.bin ({len(final_bin)} bytes including 5-byte header)")

    print("\n--- [DECODE PHASE] ---")
    decoded_text, used_strat, dec_ms = system.decode_file("output.bin", "restored.txt")
    print(f"Decoded with: {used_strat}")
    print(f"Decode speed: {dec_ms:.3f} ms")

    if orig_text == decoded_text:
        print("✅ LOSSLESS VERIFICATION PASSED: Decoded text matches exactly.")
    else:
        print("❌ LOSSLESS VERIFICATION FAILED: Texts do not match.")
