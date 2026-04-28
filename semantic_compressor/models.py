from dataclasses import dataclass

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
