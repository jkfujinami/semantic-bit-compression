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

def encode_leb128(vals: List[int]) -> bytes:
    out = bytearray()
    for val in vals:
        while True:
            byte = val & 0x7f
            val >>= 7
            if val == 0:
                out.append(byte)
                break
            else:
                out.append(byte | 0x80)
    return bytes(out)

def decode_leb128(data: bytes, count: int) -> List[int]:
    vals = []
    idx = 0
    for _ in range(count):
        val = 0
        shift = 0
        while True:
            byte = data[idx]
            idx += 1
            val |= (byte & 0x7f) << shift
            if not (byte & 0x80):
                break
            shift += 7
        vals.append(val)
    return vals

def decode_leb128_all(data: bytes) -> List[int]:
    vals = []
    idx = 0
    while idx < len(data):
        val = 0
        shift = 0
        while True:
            byte = data[idx]
            idx += 1
            val |= (byte & 0x7f) << shift
            if not (byte & 0x80):
                break
            shift += 7
        vals.append(val)
    return vals

class BwtMtfTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 5
    @property
    def name(self) -> str: return "BWT + MTF + VLQ + Zlib"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        
        # 1. Local Frequency Remapping
        from collections import Counter
        freqs = Counter(token_ids)
        unique_tokens = [t for t, c in freqs.most_common()]
        unique_count = len(unique_tokens)
        
        token_to_local = {t: i for i, t in enumerate(unique_tokens)}
        local_ids = [token_to_local[t] for t in token_ids]
        
        # 2. BWT
        n = len(local_ids)
        rotations = [local_ids[i:] + local_ids[:i] for i in range(n)]
        rotations.sort()
        last_col = [rot[-1] for rot in rotations]
        primary_index = rotations.index(local_ids)
        
        # 3. MTF
        mtf_list = list(range(unique_count))
        mtf_out = []
        for lid in last_col:
            idx = mtf_list.index(lid)
            mtf_out.append(idx)
            mtf_list.insert(0, mtf_list.pop(idx))
            
        # 4. LEB128
        leb_bytes = encode_leb128(mtf_out)
        
        # 5. Pack Header and Compress
        import struct
        header1 = struct.pack('<HI', unique_count, primary_index)
        header2 = pack_tokens(unique_tokens, bits_per_id)
        zlib_payload = zlib.compress(leb_bytes, level=9)
        
        return header1 + header2 + zlib_payload

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        if token_count == 0: return ""
        import struct
        
        # 1. Unpack Header
        unique_count, primary_index = struct.unpack('<HI', payload[:6])
        mapping_bits_len = (unique_count * bits_per_id + 7) // 8
        header2_bytes = payload[6:6+mapping_bits_len]
        unique_tokens = unpack_tokens(header2_bytes, bits_per_id, unique_count)
        
        # 2. Decompress Zlib
        zlib_payload = payload[6+mapping_bits_len:]
        leb_bytes = zlib.decompress(zlib_payload)
        
        # 3. Decode LEB128
        mtf_out = decode_leb128(leb_bytes, token_count)
        
        # 4. Inverse MTF
        mtf_list = list(range(unique_count))
        last_col = []
        for idx in mtf_out:
            lid = mtf_list.pop(idx)
            last_col.append(lid)
            mtf_list.insert(0, lid)
            
        # 5. Inverse BWT
        table = [(char, i) for i, char in enumerate(last_col)]
        table.sort(key=lambda x: x[0])
        
        local_ids = []
        curr = primary_index
        for _ in range(token_count):
            local_ids.append(table[curr][0])
            curr = table[curr][1]
            
        # 6. Reverse Local Frequency Remapping
        original_ids = [unique_tokens[lid] for lid in local_ids]
        
        return tokenizer.decode(original_ids)

class FenwickTree:
    def __init__(self, size: int):
        self.tree = [0] * (size + 1)
        self.size = size

    def add(self, i: int, delta: int):
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

class FastMTF:
    def __init__(self, vocab_size: int, max_tokens: int):
        self.size = vocab_size + max_tokens
        self.bit = FenwickTree(self.size)
        self.pos = [0] * vocab_size
        self.pos_to_token = [-1] * (self.size + 1)
        self.current_front = max_tokens
        
        # O(N) initialization of Fenwick Tree
        for v in range(vocab_size):
            p = max_tokens + v + 1
            self.pos[v] = p
            self.pos_to_token[p] = v
            self.bit.tree[p] = 1
            
        for i in range(1, self.size + 1):
            parent = i + (i & (-i))
            if parent <= self.size:
                self.bit.tree[parent] += self.bit.tree[i]

    def encode(self, token: int) -> int:
        p = self.pos[token]
        rank = self.bit.query(p) - 1
        self.bit.add(p, -1)
        self.pos[token] = self.current_front
        self.pos_to_token[self.current_front] = token
        self.bit.add(self.current_front, 1)
        self.current_front -= 1
        return rank

    def decode(self, rank: int) -> int:
        target = rank + 1
        p = 0
        step = 1 << (self.size.bit_length() - 1)
        while step > 0:
            next_p = p + step
            if next_p <= self.size and self.bit.tree[next_p] < target:
                p = next_p
                target -= self.bit.tree[p]
            step >>= 1
        p += 1
        token = self.pos_to_token[p]
        self.bit.add(p, -1)
        self.pos[token] = self.current_front
        self.pos_to_token[self.current_front] = token
        self.bit.add(self.current_front, 1)
        self.current_front -= 1
        return token

class HeaderlessBwtMtfTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 6
    @property
    def name(self) -> str: return "Headerless BWT+MTF (Pure)"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        
        vocab_size = 1 << bits_per_id
        n = len(token_ids)
        
        # 1. BWT
        rotations = [token_ids[i:] + token_ids[:i] for i in range(n)]
        rotations.sort()
        last_col = [rot[-1] for rot in rotations]
        primary_index = rotations.index(token_ids)
        
        # 2. MTF via Fenwick Tree
        mtf = FastMTF(vocab_size, n)
        mtf_out = [mtf.encode(t) for t in last_col]
            
        # 3. LEB128
        leb_bytes = encode_leb128(mtf_out)
        
        # 4. Pack Header (Only primary_index, 4 bytes)
        import struct
        header1 = struct.pack('<I', primary_index)
        zlib_payload = zlib.compress(leb_bytes, level=9)
        
        return header1 + zlib_payload

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        if token_count == 0: return ""
        import struct
        
        vocab_size = 1 << bits_per_id
        
        # 1. Unpack Header
        primary_index = struct.unpack('<I', payload[:4])[0]
        
        # 2. Decompress Zlib
        zlib_payload = payload[4:]
        leb_bytes = zlib.decompress(zlib_payload)
        
        # 3. Decode LEB128
        mtf_out = decode_leb128(leb_bytes, token_count)
        
        # 4. Inverse MTF via Fenwick Tree
        mtf = FastMTF(vocab_size, token_count)
        last_col = [mtf.decode(rank) for rank in mtf_out]
            
        # 5. Inverse BWT
        table = [(char, i) for i, char in enumerate(last_col)]
        table.sort(key=lambda x: x[0])
        
        original_ids = []
        curr = primary_index
        for _ in range(token_count):
            original_ids.append(table[curr][0])
            curr = table[curr][1]
        
        return tokenizer.decode(original_ids)

class ChimeraMtfTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 7
    @property
    def name(self) -> str: return "Chimera-MTF (Context-Aware LZ77)"

    def _get_params(self):
        return {
            "cmd_bits": 1,
            "rle_count_bits": 8,
            "lz_dist_bits": 12,
            "lz_len_bits": 8,
            "avg_lit_bits": 6
        }

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        writer = BitWriter()
        p = self._get_params()
        vocab_size = 1 << bits_per_id
        mtf = FastMTF(vocab_size, len(token_ids))
        
        avg_lit_bits = p["avg_lit_bits"]
        cost_lz77 = 1 + p["cmd_bits"] + p["lz_dist_bits"] + p["lz_len_bits"]
        
        i = 0
        window_size = (1 << p["lz_dist_bits"]) - 1
        max_lz_len = (1 << p["lz_len_bits"]) - 1
        max_rle_len = (1 << p["rle_count_bits"]) - 1
        
        while i < len(token_ids):
            # RLE
            rle_len = 1
            while i + rle_len < len(token_ids) and token_ids[i + rle_len] == token_ids[i] and rle_len < max_rle_len:
                rle_len += 1
            cost_rle = 1 + p["cmd_bits"] + avg_lit_bits + p["rle_count_bits"]
            rle_savings = (rle_len * avg_lit_bits) - cost_rle
            
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
            lz_savings = (lz_len * avg_lit_bits) - cost_lz77
            
            if max(rle_savings, lz_savings) > 0:
                if rle_savings >= lz_savings:
                    writer.write(1, 1) # Command
                    writer.write(0, p["cmd_bits"]) # RLE
                    rank = mtf.encode(token_ids[i])
                    writer.write_elias_gamma(rank)
                    writer.write(rle_len, p["rle_count_bits"])
                    for _ in range(rle_len - 1):
                        mtf.encode(token_ids[i])
                    i += rle_len
                else:
                    writer.write(1, 1) # Command
                    writer.write(1, p["cmd_bits"]) # LZ77
                    writer.write(lz_dist, p["lz_dist_bits"])
                    writer.write(lz_len, p["lz_len_bits"])
                    for k in range(lz_len):
                        mtf.encode(token_ids[i + k])
                    i += lz_len
            else:
                writer.write(0, 1) # Literal
                rank = mtf.encode(token_ids[i])
                writer.write_elias_gamma(rank)
                i += 1
                
        return writer.get_bytes()
        
    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        if token_count == 0: return ""
        from .models import BitReader
        reader = BitReader(payload)
        p = self._get_params()
        vocab_size = 1 << bits_per_id
        mtf = FastMTF(vocab_size, token_count)
        ids = []
        
        while len(ids) < token_count:
            is_cmd = reader.read(1)
            if is_cmd == 1:
                cmd = reader.read(p["cmd_bits"])
                if cmd == 0: # RLE
                    rank = reader.read_elias_gamma()
                    rle_count = reader.read(p["rle_count_bits"])
                    token = mtf.decode(rank)
                    ids.append(token)
                    for _ in range(rle_count - 1):
                        ids.append(token)
                        mtf.encode(token)
                elif cmd == 1: # LZ77
                    dist = reader.read(p["lz_dist_bits"])
                    length = reader.read(p["lz_len_bits"])
                    start_idx = len(ids) - dist
                    for k in range(length):
                        token = ids[start_idx + k]
                        ids.append(token)
                        mtf.encode(token)
            else: # Literal
                rank = reader.read_elias_gamma()
                token = mtf.decode(rank)
                ids.append(token)
                
        return tokenizer.decode(ids)

class BwtChimeraTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 8
    @property
    def name(self) -> str: return "BWT + MTF + Chimera (RLE+LZ77+EliasGamma)"

    def _get_params(self):
        return {
            "cmd_bits": 1,
            "rle_count_bits": 10,
            "lz_dist_bits": 12,
            "lz_len_bits": 8,
            "avg_lit_bits": 4
        }

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        import struct
        
        vocab_size = 1 << bits_per_id
        n = len(token_ids)
        
        # 1. BWT
        rotations = [token_ids[i:] + token_ids[:i] for i in range(n)]
        rotations.sort()
        last_col = [rot[-1] for rot in rotations]
        primary_index = rotations.index(token_ids)
        
        # 2. MTF via Fenwick Tree
        mtf = FastMTF(vocab_size, n)
        ranks = [mtf.encode(t) for t in last_col]
        
        # 3. Chimera-style RLE + LZ77 on rank stream with Elias Gamma
        writer = BitWriter()
        p = self._get_params()
        avg_lit_bits = p["avg_lit_bits"]
        cost_lz77 = 1 + p["cmd_bits"] + p["lz_dist_bits"] + p["lz_len_bits"]
        
        window_size = (1 << p["lz_dist_bits"]) - 1
        max_lz_len = (1 << p["lz_len_bits"]) - 1
        max_rle_len = (1 << p["rle_count_bits"]) - 1
        
        i = 0
        while i < n:
            # RLE detection
            rle_len = 1
            while i + rle_len < n and ranks[i + rle_len] == ranks[i] and rle_len < max_rle_len:
                rle_len += 1
            cost_rle = 1 + p["cmd_bits"] + avg_lit_bits + p["rle_count_bits"]
            rle_savings = (rle_len * (1 + avg_lit_bits)) - cost_rle
            
            # LZ77 detection
            lz_len = 0
            lz_dist = 0
            start_w = max(0, i - window_size)
            for j in range(start_w, i):
                l = 0
                while i + l < n and ranks[j + l] == ranks[i + l] and l < max_lz_len:
                    l += 1
                if l > lz_len:
                    lz_len = l
                    lz_dist = i - j
            lz_savings = (lz_len * (1 + avg_lit_bits)) - cost_lz77
            
            if max(rle_savings, lz_savings) > 0:
                if rle_savings >= lz_savings:
                    writer.write(1, 1)  # Command flag
                    writer.write(0, p["cmd_bits"])  # RLE
                    writer.write_elias_gamma(ranks[i])
                    writer.write(rle_len, p["rle_count_bits"])
                    i += rle_len
                else:
                    writer.write(1, 1)  # Command flag
                    writer.write(1, p["cmd_bits"])  # LZ77
                    writer.write(lz_dist, p["lz_dist_bits"])
                    writer.write(lz_len, p["lz_len_bits"])
                    i += lz_len
            else:
                writer.write(0, 1)  # Literal flag
                writer.write_elias_gamma(ranks[i])
                i += 1
                
        # 4. Pack: primary_index (4 bytes) + compressed bitstream
        header = struct.pack('<I', primary_index)
        return header + writer.get_bytes()
        
    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        if token_count == 0: return ""
        import struct
        
        vocab_size = 1 << bits_per_id
        
        # 1. Unpack header
        primary_index = struct.unpack('<I', payload[:4])[0]
        
        # 2. Decode Chimera bitstream → rank stream
        reader = BitReader(payload[4:])
        p = self._get_params()
        ranks = []
        
        while len(ranks) < token_count:
            is_cmd = reader.read(1)
            if is_cmd == 1:
                cmd = reader.read(p["cmd_bits"])
                if cmd == 0:  # RLE
                    rank = reader.read_elias_gamma()
                    rle_count = reader.read(p["rle_count_bits"])
                    ranks.extend([rank] * rle_count)
                elif cmd == 1:  # LZ77
                    dist = reader.read(p["lz_dist_bits"])
                    length = reader.read(p["lz_len_bits"])
                    start_idx = len(ranks) - dist
                    for k in range(length):
                        ranks.append(ranks[start_idx + k])
            else:  # Literal
                rank = reader.read_elias_gamma()
                ranks.append(rank)
        
        # 3. Inverse MTF via Fenwick Tree
        mtf = FastMTF(vocab_size, token_count)
        last_col = [mtf.decode(r) for r in ranks]
            
        # 4. Inverse BWT
        table = [(char, i) for i, char in enumerate(last_col)]
        table.sort(key=lambda x: x[0])
        
        original_ids = []
        curr = primary_index
        for _ in range(token_count):
            original_ids.append(table[curr][0])
            curr = table[curr][1]
        
        return tokenizer.decode(original_ids)

class BwtRleZeroTokenStrategy(ICompressionStrategy):
    @property
    def strategy_id(self) -> int: return 9
    @property
    def name(self) -> str: return "BWT + MTF + RLE-Zero + Zlib"

    def encode(self, text: str, token_ids: List[int], bits_per_id: int) -> bytes:
        if not token_ids: return b""
        import struct
        vocab_size = 1 << bits_per_id
        n = len(token_ids)

        # 1. BWT
        rotations = [token_ids[i:] + token_ids[:i] for i in range(n)]
        rotations.sort()
        last_col = [rot[-1] for rot in rotations]
        primary_index = rotations.index(token_ids)

        # 2. MTF via Fenwick Tree
        mtf = FastMTF(vocab_size, n)
        ranks = [mtf.encode(t) for t in last_col]

        # 3. RLE-Zero (bzip2 RUNA/RUNB bijective base-2)
        symbols = []
        i = 0
        while i < n:
            if ranks[i] == 0:
                zero_count = 0
                while i < n and ranks[i] == 0:
                    zero_count += 1
                    i += 1
                while zero_count > 0:
                    if zero_count % 2 == 1:
                        symbols.append(0)  # RUNA
                        zero_count = (zero_count - 1) // 2
                    else:
                        symbols.append(1)  # RUNB
                        zero_count = (zero_count - 2) // 2
            else:
                symbols.append(ranks[i] + 1)  # shift +1
                i += 1

        # 4. LEB128 + Zlib
        leb_bytes = encode_leb128(symbols)
        header = struct.pack('<I', primary_index)
        return header + zlib.compress(leb_bytes, level=9)

    def decode(self, payload: bytes, tokenizer, bits_per_id: int, token_count: int) -> str:
        if token_count == 0: return ""
        import struct
        vocab_size = 1 << bits_per_id

        primary_index = struct.unpack('<I', payload[:4])[0]
        leb_bytes = zlib.decompress(payload[4:])
        symbols = decode_leb128_all(leb_bytes)

        # Decode RLE-Zero
        ranks = []
        i = 0
        while len(ranks) < token_count:
            if i < len(symbols) and symbols[i] <= 1:
                power = 0
                zero_count = 0
                while i < len(symbols) and symbols[i] <= 1:
                    zero_count += (symbols[i] + 1) * (1 << power)
                    power += 1
                    i += 1
                ranks.extend([0] * zero_count)
            else:
                ranks.append(symbols[i] - 1)
                i += 1

        # Inverse MTF
        mtf = FastMTF(vocab_size, token_count)
        last_col = [mtf.decode(r) for r in ranks]

        # Inverse BWT
        table = [(char, i) for i, char in enumerate(last_col)]
        table.sort(key=lambda x: x[0])
        original_ids = []
        curr = primary_index
        for _ in range(token_count):
            original_ids.append(table[curr][0])
            curr = table[curr][1]
        return tokenizer.decode(original_ids)
