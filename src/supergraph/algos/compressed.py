
import numpy as np

def pack_vbyte(arr: np.ndarray) -> bytes:
    if len(arr) == 0:
        return b""
    
    
    out = bytearray()
    for val in arr:
        val = int(val)
        while val >= 128:
            out.append((val & 0x7f) | 0x80)
            val >>= 7
        out.append(val & 0x7f)
    return bytes(out)

def unpack_vbyte(data: bytes, count: int) -> np.ndarray:
    res = np.zeros(count, dtype=np.uint32)
    if not data:
        return res
        
    pos = 0
    for i in range(count):
        val = 0
        shift = 0
        while pos < len(data):
            b = data[pos]
            pos += 1
            val |= (b & 0x7f) << shift
            if not (b & 0x80):
                break
            shift += 7
        res[i] = val
    return res

def pack_delta_vbyte(arr: np.ndarray) -> bytes:
    if len(arr) == 0:
        return b""
    deltas = np.diff(arr, prepend=0)
    return pack_vbyte(deltas)

def unpack_delta_vbyte(data: bytes, count: int) -> np.ndarray:
    if count == 0:
        return np.array([], dtype=np.uint32)
    deltas = unpack_vbyte(data, count)
    return np.cumsum(deltas).astype(np.uint32)
