import torch
import torch.nn as nn
import math
text = "I love learning AI I love learning AI"
words = text.split()
vocab = sorted(set(words))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}
print("Vocabulary:")
print(word_to_id)
ids = [word_to_id[word] for word in words]
inputs = []
targets = []
sequence_length = 3
for i in range(len(ids) - sequence_length):
    input_seq = ids[i:i + sequence_length]
    target_seq = ids[i + 1:i + sequence_length + 1]
    inputs.append(input_seq)
    targets.append(target_seq)
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
print("\nInputs:")
print(inputs)
print("\nTargets:")
print(targets)
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        self.Wq = nn.Linear(
            d_model,
            d_model
        )
        self.Wk = nn.Linear(
            d_model,
            d_model
        )
        self.Wv = nn.Linear(
            d_model,
            d_model
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )
        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )
    def forward(self, x):
        x = self.embedding(x)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_model)
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(
            mask,
            float("-inf")
        )
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )
        x = attention_weights @ V
        x = self.ffn(x)
        logits = self.output_layer(x)
        return logits
model = TinyTransformer(
    vocab_size=len(vocab),
    d_model=16
)
print("\nModel created!")
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
epochs = 1000
for epoch in range(epochs):
    predictions = model(inputs)
    loss = loss_function(
        predictions.reshape(-1, len(vocab)),
        targets.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.4f}"
        )
print("\n==============================")
print("TEST")

model.eval()
with torch.no_grad():
    test_words = [
        "I",
        "love",
        "learning"
    ]
    test_ids = torch.tensor([
        [word_to_id[word] for word in test_words]
    ])
    output = model(test_ids)
    last_logits = output[0, -1]
    predicted_id = torch.argmax(
        last_logits
    ).item()
    predicted_word = id_to_word[predicted_id]
    print("Input:")
    print(test_words)
    print("\nPredicted next word:")
    print(predicted_word)