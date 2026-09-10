import torch
import torch.nn as nn
import math
#simple input
sentence="I love learning AI"
tokens=sentence.split()
print("Tokens:",tokens)

##Token ids
vocab={
    "I":0,"love":1,"learning":2,"AI":3
}
token_ids=torch.tensor([
    vocab[token] for token in tokens
])
print("\nToken_ids:",token_ids)

##Embeddings
embedding=nn.Embedding(
    num_embeddings=4,
    embedding_dim=8
)
x=embedding(token_ids)
print("\nEmbedding shape:",x.shape)
print("\nEmbedding:",x)

##Self attention
## create query,key,values
d_model=8
wq=nn.Linear(d_model,d_model)
wk=nn.Linear(d_model,d_model)
wv=nn.Linear(d_model,d_model)
Q=wq(x)
K=wk(x)
V=wv(x)
print("\nQ shape:", Q.shape) 
print("K shape:", K.shape) 
print("V shape:", V.shape)

##attention scores
scores=Q @ K .transpose(-2,-1)
scores=scores/math.sqrt(d_model)
print("\n Attention Scores:",scores)

##casual mask 
sequence_length=len(tokens)
mask=torch.triu(
    torch.ones(sequence_length,sequence_length),
    diagonal=1
).bool()
scores=scores.masked_fill(mask,float("-inf"))
print("\ncasual masked scores:",scores)


##softmax
attention_weights=torch.softmax(
    scores,dim=-1
)
print("\n Attention weights:",attention_weights)

## attention output
attention_output=attention_weights @ V
print("\n Attention output:",attention_output)

##feed forward network
feed_forward=nn.Sequential(
    nn.Linear(d_model,16),
    nn.ReLU(),
    nn.Linear(16,d_model)
)
output=feed_forward(attention_output)
print("\n Final transformer output:",output)
print("\nShape:",output.shape)