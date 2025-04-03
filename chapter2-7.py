import torch
input_ids = torch.tensor([2,3,5,1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print('embedding layer')
print(embedding_layer.weight)

print('idをベクトル変換')
print(embedding_layer(input_ids))

pos_embedding_layer = torch.nn.Embedding(4, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(4))
print('位置ベクトル変換')
print(pos_embeddings)
