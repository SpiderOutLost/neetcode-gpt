from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        arr = set(text)
        arr_sort = sorted(arr)
        stoi = {key: index for index, key in enumerate(arr_sort)}
        itos = {index: key for key, index in stoi.items()}
        return stoi, itos
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        encoder = []
        for ch in text:
            encoder.append(stoi[ch])
        return encoder

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        decoder = []
        for num in ids:
            decoder.append(itos[num])
        return "".join(decoder)
