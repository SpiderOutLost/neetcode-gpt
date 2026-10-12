from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        token = []
        for num in numbers:
            num = str(num)
            num_tokens = []
            pos = 0
            while pos < len(num):
                L = len(num) - pos
                while L>0:
                    if num[pos:pos+L] in vocab.keys():
                        num_tokens.append(num[pos:pos+L])
                        break
                    else: L-=1
                pos += L
            token.append(num_tokens)
        return token
    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        token_length = 0
        num_tokens = 0
        pos = 0
        while pos < len(text):
            L = len(text) - pos
            while L>0:
                if text[pos:pos+L] in vocab.keys():
                    num_tokens += 1
                    break
                else: L-=1
            pos += L
        token_length += num_tokens
        return token_length

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens_count = self.count_tokens(text,vocab)
        return round(tokens_count/len(text.split()), 4)