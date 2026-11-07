"""Character-level tokenizer: every unique character is a token."""


class CharTokenizer:
    def __init__(self, text=None, chars=None):
        if chars is None:
            chars = sorted(set(text))
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    tok = CharTokenizer(text="hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    print("round-trip OK:", ids, "->", tok.decode(ids))
