import zlib
from abc import ABC, abstractmethod
from typing import List
from .models import BitWriter, BitReader

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
            # RLE
            rle_len = 1
            while i + rle_len < len(token_ids) and token_ids[i + rle_len] == token_ids[i] and rle_len < max_rle_len:
                rle_len += 1
            rle_savings = (rle_len * bits_per_id) - cost_rle
            
            # LZ77
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
