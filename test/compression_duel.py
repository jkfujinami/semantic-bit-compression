import time
from abc import ABC, abstractmethod
import zlib
import os
from dataclasses import dataclass
from typing import List, Tuple
from tokenizers import Tokenizer

# ==========================================
# 1. Models (データモデル)
# ==========================================
@dataclass
class CompressionResult:
    strategy_name: str
    payload: bytes
    _explicit_bits: int = -1
    execution_time_ms: float = 0.0
    
    @property
    def bits(self) -> int:
        if self._explicit_bits >= 0:
            return self._explicit_bits
        return len(self.payload) * 8

# ==========================================
# 2. Utilities (単一責任: ビットの詰め込み)
# ==========================================
class BitPacker:
    """18bitのトークンIDを隙間なく連続したバイト配列(バイナリ)にパッキングする"""
    @staticmethod
    def pack(ids: List[int], bits_per_id: int) -> bytes:
        if not ids: return b""
        bit_stream = 0
        total_bits = 0
        for token_id in ids:
            bit_stream = (bit_stream << bits_per_id) | token_id
            total_bits += bits_per_id
        
        byte_length = (total_bits + 7) // 8
        return bit_stream.to_bytes(byte_length, byteorder='big')

# ==========================================
# 3. Interfaces (依存性逆転 / オープン・クローズド)
# ==========================================
class ICompressionStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        pass

# ==========================================
# 4. Strategies (各圧縮アルゴリズムの実装)
# ==========================================
class RawUtf8Strategy(ICompressionStrategy):
    @property
    def name(self) -> str: return "Raw UTF-8 (Fallback)"
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        start = time.perf_counter()
        payload = text.encode('utf-8')
        end = time.perf_counter()
        return CompressionResult(self.name, payload, execution_time_ms=(end-start)*1000)

class ZlibRawUtf8Strategy(ICompressionStrategy):
    @property
    def name(self) -> str: return "Zlib (Deflate) on Raw UTF-8"
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        start = time.perf_counter()
        payload = zlib.compress(text.encode('utf-8'), level=9)
        end = time.perf_counter()
        return CompressionResult(self.name, payload, execution_time_ms=(end-start)*1000)

class RawTokenStrategy(ICompressionStrategy):
    @property
    def name(self) -> str: return "Raw Tokens (18-bit native)"
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        start = time.perf_counter()
        payload = BitPacker.pack(token_ids, bits_per_id)
        end = time.perf_counter()
        return CompressionResult(self.name, payload, execution_time_ms=(end-start)*1000)

class ZlibTokenStrategy(ICompressionStrategy):
    @property
    def name(self) -> str: return "Zlib (Deflate) on Tokens"
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        start = time.perf_counter()
        packed = BitPacker.pack(token_ids, bits_per_id)
        payload = zlib.compress(packed, level=9)
        end = time.perf_counter()
        return CompressionResult(self.name, payload, execution_time_ms=(end-start)*1000)

class ChimeraTokenStrategy(ICompressionStrategy):
    @property
    def name(self) -> str: return "Chimera (Escape-based RLE + LZ77)"
    def compress(self, text: str, token_ids: List[int], bits_per_id: int) -> CompressionResult:
        start = time.perf_counter()
        if not token_ids:
            end = time.perf_counter()
            return CompressionResult(self.name, b"", execution_time_ms=(end-start)*1000)
            
        ESCAPE_ID = (1 << bits_per_id) - 1
        
        cmd_bits = 2       
        rle_count_bits = 8 
        lz_dist_bits = 12  
        lz_len_bits = 8    
        
        cost_literal = bits_per_id
        cost_escape_literal = bits_per_id + cmd_bits
        cost_rle = bits_per_id + cmd_bits + bits_per_id + rle_count_bits
        cost_lz77 = bits_per_id + cmd_bits + lz_dist_bits + lz_len_bits
        
        i = 0
        total_bits = 0
        
        window_size = (1 << lz_dist_bits) - 1
        max_lz_len = (1 << lz_len_bits) - 1
        max_rle_len = (1 << rle_count_bits) - 1
        
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
                    total_bits += cost_rle
                    i += rle_len
                else:
                    total_bits += cost_lz77
                    i += lz_len
            else:
                if token_ids[i] == ESCAPE_ID:
                    total_bits += cost_escape_literal
                else:
                    total_bits += cost_literal
                i += 1
                
        byte_length = (total_bits + 7) // 8
        dummy_payload = bytes(byte_length)
        
        end = time.perf_counter()
        return CompressionResult(self.name, dummy_payload, total_bits, execution_time_ms=(end-start)*1000)

# ==========================================
# 5. Context (戦略の選択とルール適用)
# ==========================================
class SmartCompressor:
    def __init__(self, strategies: List[ICompressionStrategy]):
        self.strategies = strategies

    def execute(self, text: str, token_ids: List[int], bits_per_id: int) -> Tuple[CompressionResult, List[CompressionResult]]:
        results = [s.compress(text, token_ids, bits_per_id) for s in self.strategies]
        
        raw_res = next(r for r in results if r.strategy_name == "Raw UTF-8 (Fallback)")
        packed_res = next(r for r in results if r.strategy_name == "Raw Tokens (18-bit native)")
        
        default_res = packed_res if packed_res.bits <= raw_res.bits else raw_res
        
        best_res = default_res
        for r in results:
            if r.strategy_name not in ["Raw UTF-8 (Fallback)", "Raw Tokens (18-bit native)"]:
                if r.bits < best_res.bits:
                    best_res = r
                    
        return best_res, results

# ==========================================
# Main Execution
# ==========================================
def main():
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("Error: tokenizer.json not found.")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bits_per_id = 18

    # 先ほどのチャットログ
    chat_log = """08:32 焼肉 今回の制作はgoの勉強も兼ねているし、車輪の再発明にも意味はあります💢
08:36 焼肉 蓮舫かよ‼️
08:38 ふじゃ 勉強なら手書き....
09:28 jelly cat PowerShellにC#やらpythonやらを埋め込んで、ターミナルで実行するだけで動くフォームアプリケーション作ってます。Win端末なら標準で入ってますし、ファイル/Excel操作も常駐アプリ化もお手の物なので職場でも重宝してます。
09:32 NWマン AI時代にコーディング学ぶのってどうすりゃいいんでしょうね。
Vibe coding的に雑に投げてると学びにならないけど、かと言って全部手書きというのも中々…。

動くアプリ作って、動かない場合のトラシューを丸投げしないで主体で切り分ける&デザインパターンみたいなコーディングより上のレイヤーの知見を学ぶとかの方が良いんだろうか(考える)
09:33 NWマン AI、上手く使えばアクティブラーニング的に使えると思ってるのだけどうまい言語化が思いつかない……
09:41 かぼす 興味深いです！ Web操作とPCローカル作業も横断できるかんじでしょうか?
RPA的にタスク完了させるツール開発したいです
09:57 落武者 powershellにPythonなどを埋め込むってどういうことでしょうか？全くイメージが湧かず…
10:00 panai おもしろそうー！私もつくろう
10:04 panai AI に仕組み聞きました。そんなことできるなんてしりませんでした。

10:06 jelly cat 方法次第でできると思います。自分の知識ではSelenium操作程度の自動化知識しかないのですが、設定したブラウザ動作で取得した情報を変数に入れてあれこれするのは可能です。
10:07 panai claude って、ハック的なコンプラ厳しいやつあんまり強力してくれないでしょ
10:08 panai 強力→協力
10:22 jelly cat あれ
10:23 jelly cat なんかリプライ送ろうとすると消されます
10:23 さかもと NGワードが入ってるからじゃないですか
10:24 jelly cat pythonはヒアドキュメントで定義したスクリプトをpythonの実行ファイルに引数として渡してあげるだけです。ヒアドキュメント内や実行の戻り値をPowerShellの変数で共有管理すればPowerShellの処理と連携可能です（力技ですが）
10:24 jelly catがメッセージの送信を取り消しました
10:25 jelly cat C#コードはヒアドキュメントでテキストとして変数に格納し、Add-Typeで読み込むと. NETアセンブリとして組み込まれて利用可能になります。
10:25 jelly cat . NETかな
10:45 panai 言葉でプログラミングしてるだけなので、まだまだそういった知識がないので、勉強になります。mdファイルのことも知りませんでしたからね。
12:09 レモンスカッシュ pythonは別途インストール必要ですよね？(?)
他の人に配布する時
12:10 未学者 Pyinstallerで実行ファイルにしたらダブルクリックで起動できませんでしたっけ
12:12 落武者 そういうイメージなんですね。ありがとうございます。
12:12 レモンスカッシュ exeにして配布ですね(了解)
12:13 レモンスカッシュ c#で配布するのはいつもしているので、
pythonで配布できる可能性に反応しました
12:40 jelly cat pythonはインストール（モジュール含め）が必要です。実行をPowerShell内でpyの中身を内部生成して、一時ファイル書き出しやpyファイルの設置なしで直接実行して、変数でやり取りできるというだけですね。
Tkinterなどのpython GUIよりもWindows Form+C#のほうがより使いやすいやすいUIが利用できると思うので、処理だけpythonに任せたい場合に使っています。
12:43 jelly cat PowerShellの良いところはスクリプトをコピペするだけで実行できるのでC#の埋め込みならインストールなしで配布して、環境依存無しで直ぐに実行できるところですかね。画像などのファイルはBase64で埋め込めば全てテキスト情報だけで完結します。
12:49 Fujis 自分はquizのskill作って、mdに吐き出させるようにしてます。/quiz xxxで自動的にカリキュラム作成みたいな。カリキュラムの最後に設問が来るようにプロンプトを用意し、知識が定着しているか確認できるようにしました。
13:38 レモンスカッシュ いいとこ取りですね、ありがとうございます
15:49 Gump 現状でフロントやらせるならコーデックスかクロードならどっちですか？
16:02 NWマン なるほど、Quiz形式でやらせるんですな"""

    texts = [
        ("Short Japanese", "こんにちは。"),
        ("100M Yen", "100000000円欲しいです。"),
        ("Chat Log", chat_log)
    ]

    compressor = SmartCompressor([
        RawUtf8Strategy(),
        ZlibRawUtf8Strategy(),
        RawTokenStrategy(),
        ZlibTokenStrategy(),
        ChimeraTokenStrategy()
    ])

    for name, text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        best_res, all_res = compressor.execute(text, ids, bits_per_id)
        
        raw_res = next(r for r in all_res if r.strategy_name == "Raw UTF-8 (Fallback)")
        raw_bits = raw_res.bits
        
        print(f"\n=== Case: {name} ===")
        print(f"Token Count: {len(ids)}")
        for r in all_res:
            indicator = "★ BEST" if r == best_res else "      "
            ratio = (r.bits / raw_bits) * 100
            print(f"{indicator} | {r.bits:>6} bits ({ratio:>5.1f}%) | {r.execution_time_ms:>6.3f} ms | {r.strategy_name}")

if __name__ == "__main__":
    main()
