import torch
from gpt_dataset_v1 import create_dataloader_v1
import urllib.request

max_length = 4

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
raw_text = urllib.request.urlopen(url).read().decode("utf-8")  # デコードを追加

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs")
print(inputs)
print("Token IDs shape")
print(inputs.shape)

# embedding layer
vocab_size = 50257
output_dim = 256
torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print("Token Embeddings")
print(token_embeddings)

# positional embedding layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional Embeddings")
print(pos_embeddings)

# add positional embedding
input_embeddings = token_embeddings + pos_embeddings
print("Input Embeddings")
print(input_embeddings)