import urllib.request
import re

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
raw_text = urllib.request.urlopen(url).read().decode("utf-8")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab_size = len(all_tokens)
print(f"Vocabulary size: {vocab_size}")
vocab = {token: index for index, token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
      self.str_to_int = vocab
      self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text):
      preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      preprocessed = [item.strip() for item in preprocessed if item.strip()]
      preprocessed = ["<|unk|>" if token not in self.str_to_int else token for token in preprocessed]
      ids = [self.str_to_int[token] for token in preprocessed]
      return ids
    
    def decode(self, ids):
      text = " ".join([self.int_to_str[index] for index in ids])
      text = re.sub(r'\s+([.,:;?_!"()\'])', r'\1', text)
      return text

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, Do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print(text)

print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))