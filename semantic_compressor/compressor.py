import struct
import time
from typing import Optional, List, Tuple
from tokenizers import Tokenizer
from .models import EncodeResult
from .strategies import (
    RawUtf8Strategy,
    ZlibRawUtf8Strategy,
    RawTokenStrategy,
    ZlibTokenStrategy,
    ChimeraTokenStrategy,
    BwtMtfTokenStrategy,
    HeaderlessBwtMtfTokenStrategy,
    ChimeraMtfTokenStrategy
)

class SemanticCompressor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bits_per_id = 18
        self.strategies = [
            RawUtf8Strategy(),
            ZlibRawUtf8Strategy(),
            RawTokenStrategy(),
            ZlibTokenStrategy(),
            ChimeraTokenStrategy(),
            BwtMtfTokenStrategy(),
            HeaderlessBwtMtfTokenStrategy(),
            ChimeraMtfTokenStrategy()
        ]
        self.strat_map = {s.strategy_id: s for s in self.strategies}

    def encode(self, text: Optional[str] = None, input_filepath: Optional[str] = None, output_filepath: Optional[str] = None) -> bytes:
        """
        エンコード処理。文字列(text)またはファイル(input_filepath)を受け取り、
        圧縮されたバイナリを返します。output_filepathを指定した場合はファイルにも書き出します。
        """
        if text is None and input_filepath is None:
            raise ValueError("Either text or input_filepath must be provided.")
            
        if input_filepath:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        token_count = len(token_ids)
        
        results = []
        for strat in self.strategies:
            payload = strat.encode(text, token_ids, self.bits_per_id)
            decoded_text = strat.decode(payload, self.tokenizer, self.bits_per_id, token_count)
            
            # Lossless validation
            if text != decoded_text and strat.strategy_id != 0:
                continue
                
            results.append(EncodeResult(
                strategy_id=strat.strategy_id,
                strategy_name=strat.name,
                payload=payload,
                encode_time_ms=0,
                decode_time_ms=0
            ))
            
        # Select best fallback strategy (Raw UTF-8 is always ID 0)
        best_res = next(r for r in results if r.strategy_id == 0)
        for r in results:
            if len(r.payload) < len(best_res.payload):
                best_res = r
                
        # Header: [1 byte ID] + [4 bytes uint32 Token Count]
        header = struct.pack('<BI', best_res.strategy_id, token_count)
        final_binary = header + best_res.payload
        
        if output_filepath:
            with open(output_filepath, 'wb') as f:
                f.write(final_binary)
                
        return final_binary

    def decode(self, payload: Optional[bytes] = None, input_filepath: Optional[str] = None, output_filepath: Optional[str] = None) -> str:
        """
        デコード処理。バイナリデータ(payload)またはファイル(input_filepath)を受け取り、
        復元された文字列を返します。output_filepathを指定した場合はファイルにも書き出します。
        """
        if payload is None and input_filepath is None:
            raise ValueError("Either payload or input_filepath must be provided.")
            
        if input_filepath:
            with open(input_filepath, 'rb') as f:
                payload = f.read()
                
        strategy_id, token_count = struct.unpack('<BI', payload[:5])
        strat_payload = payload[5:]
        
        strat = self.strat_map[strategy_id]
        decoded_text = strat.decode(strat_payload, self.tokenizer, self.bits_per_id, token_count)
        
        if output_filepath:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(decoded_text)
                
        return decoded_text

    def benchmark(self, text: str) -> dict:
        """
        パフォーマンスと圧縮率を測定し、内部の詳細データを返すデバッグ用メソッド
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        token_count = len(token_ids)
        
        results = []
        for strat in self.strategies:
            t0 = time.perf_counter()
            payload = strat.encode(text, token_ids, self.bits_per_id)
            t1 = time.perf_counter()
            
            t2 = time.perf_counter()
            decoded_text = strat.decode(payload, self.tokenizer, self.bits_per_id, token_count)
            t3 = time.perf_counter()
            
            if text != decoded_text and strat.strategy_id != 0:
                continue
                
            results.append(EncodeResult(
                strategy_id=strat.strategy_id,
                strategy_name=strat.name,
                payload=payload,
                encode_time_ms=(t1 - t0) * 1000,
                decode_time_ms=(t3 - t2) * 1000
            ))
            
        best_res = next(r for r in results if r.strategy_id == 0)
        for r in results:
            if len(r.payload) < len(best_res.payload):
                best_res = r
                
        return {
            "token_count": token_count,
            "raw_size": len(text.encode('utf-8')),
            "results": results,
            "best_strategy_id": best_res.strategy_id,
            "best_strategy_name": best_res.strategy_name,
            "final_size": len(best_res.payload) + 5
        }
