from tokenizer.bpe_class import BPETokenizer

text = open("data/train.txt").read()
tokenizer = BPETokenizer(text, num_merges=300)
tokenizer.save("tokenizer/")

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"BOS: {tokenizer.bos_id}, EOS: {tokenizer.eos_id}, PAD: {tokenizer.pad_id}")

