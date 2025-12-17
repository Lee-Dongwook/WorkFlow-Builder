from tokenizer.bpe_class import BPETokenizer

text = open("data/train.txt").read()
tokenizer = BPETokenizer(text, num_merges=300)
tokenizer.save("tokenizer/")

print(f"Tokenizer saved! vocab_size: {tokenizer.vocab_size}")

