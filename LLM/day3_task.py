import math
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
text = """
Artificial intelligence is a field of computer science.
Artificial intelligence allows computers to perform intelligent tasks.
Machine learning is a part of artificial intelligence.
Machine learning allows computers to learn patterns from data.
Deep learning is a type of machine learning.
Deep learning uses neural networks with multiple layers.
Neural networks are mathematical models inspired by biological neurons.
A neural network contains weights and biases.
Training updates the weights of a neural network.
The model learns by minimizing a loss function.
The loss function measures prediction error.
Gradient descent updates model parameters to reduce loss.
Backpropagation calculates gradients through the neural network.
Transformers are neural network architectures based on attention.
Attention helps a model understand relationships between tokens.
Self attention allows each token to interact with other tokens.
Causal attention prevents a token from seeing future tokens.
A language model predicts the next token.
Large language models are trained on large datasets.
Large language models process text as tokens.
Tokens are converted into numerical token ids.
Embeddings convert token ids into vectors.
Transformers process embeddings using attention layers.
Python is a popular programming language.
Python is widely used for artificial intelligence.
Python is used for machine learning and deep learning.
PyTorch is a popular framework for building neural networks.
PyTorch provides tensors and automatic differentiation.
A transformer can be trained using PyTorch.
Training requires input data and target data.
The model produces logits for possible next tokens.
Cross entropy compares model predictions with target tokens.
The optimizer updates model parameters using gradients.
Fine tuning adapts a pretrained model to a specific task.
Instruction tuning teaches a model to follow instructions.
Human feedback can be used to improve model behavior.
DPO is a method for learning from preference data.
Alignment aims to make model behavior match intended goals.
A base model mainly learns language patterns during pretraining.
An instruct model is trained to follow human instructions.
A chat model is designed for conversational interaction.
AI can be integrated into full stack applications.
Full stack applications can use AI models through APIs.
Node.js can be used to build AI application backends.
Python can be used to build machine learning services.
AI applications combine models with software engineering.
"""
def tokenize(text):
    return re.findall(r"\b\w+\b|[.!?,]", text.lower())
tokens = tokenize(text)
print("\nTotal tokens:", len(tokens))
print("\nFirst 30 tokens:")
print(tokens[:30])
special_tokens = ["<PAD>", "<UNK>"]
vocab_words = sorted(set(tokens))
vocab = special_tokens + vocab_words
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}
vocab_size = len(vocab)
print("\nVocabulary size:", vocab_size)
token_ids = [
    word_to_id.get(token, word_to_id["<UNK>"])
    for token in tokens
]
token_ids = torch.tensor(token_ids, dtype=torch.long)
print("\nToken IDs:")
print(token_ids[:30])
split_index = int(len(token_ids) * 0.90)
train_tokens = token_ids[:split_index]
val_tokens = token_ids[split_index:]
print("\nTraining tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
BLOCK_SIZE = 8
def create_sequences(data):
    inputs = []
    targets = []
    for i in range(len(data) - BLOCK_SIZE):
        input_sequence = data[i:i + BLOCK_SIZE]
        target_sequence = data[i + 1:i + BLOCK_SIZE + 1]
        inputs.append(input_sequence)
        targets.append(target_sequence)
    return torch.stack(inputs), torch.stack(targets)
train_inputs, train_targets = create_sequences(train_tokens)
val_inputs, val_targets = create_sequences(val_tokens)
print("\nTrain input shape:")
print(train_inputs.shape)
print("\nTrain target shape:")
print(train_targets.shape)
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_length, d_model)
    def forward(self, x):
        sequence_length = x.size(1)
        positions = torch.arange(
            sequence_length,
            device=x.device
        )
        positions = positions.unsqueeze(0)
        return self.position_embedding(positions)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(
            d_model,
            d_model * 3
        )
        self.output = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(
            qkv,
            3,
            dim=-1
        )
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
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
        attention_output = attention_weights @ v
        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model
            )
        )
        return self.output(attention_output)
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x):
        return self.network(x)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model=64,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        self.position_embedding = PositionalEmbedding(
            block_size,
            d_model
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )
    def forward(self, x):
        x = (
            self.token_embedding(x)
            + self.position_embedding(x)
        )
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
model = TinyLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    d_model=64,
    num_heads=4,
    num_layers=2
).to(DEVICE)
print("\nModel created!")
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)
print("Total parameters:", total_parameters)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003
)
train_inputs = train_inputs.to(DEVICE)
train_targets = train_targets.to(DEVICE)
val_inputs = val_inputs.to(DEVICE)
val_targets = val_targets.to(DEVICE)
def calculate_loss(inputs, targets):
    logits = model(inputs)
    loss = loss_function(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )
    return loss
EPOCHS = 500
train_losses = []
val_losses = []
print("\nTRAINING STARTED")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    train_loss = calculate_loss(
        train_inputs,
        train_targets
    )
    train_loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_loss = calculate_loss(
            val_inputs,
            val_targets
        )
    train_losses.append(train_loss.item())
    val_losses.append(validation_loss.item())
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {validation_loss.item():.4f}"
        )
print("\nTRAINING COMPLETE")
plt.figure(figsize=(10, 5))
plt.plot(
    train_losses,
    label="Train Loss"
)
plt.plot(
    val_losses,
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tiny Transformer Training Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("day3_loss.png")
plt.show()
print("\nLoss graph saved as:")
print("day3_loss.png")
checkpoint = {
    "model_state_dict": model.state_dict(),
    "word_to_id": word_to_id,
    "id_to_word": id_to_word,
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE
}
torch.save(
    checkpoint,
    "day3_tiny_llm.pt"
)
print("\nModel saved as:")
print("day3_tiny_llm.pt")
def generate_text(prompt, max_new_tokens=20):
    model.eval()
    prompt_tokens = tokenize(prompt)
    ids = [
        word_to_id.get(
            token,
            word_to_id["<UNK>"]
        )
        for token in prompt_tokens
    ]
    generated_ids = ids.copy()
    for _ in range(max_new_tokens):
        context = generated_ids[-BLOCK_SIZE:]
        input_tensor = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1]
        next_token_id = torch.argmax(
            next_token_logits
        ).item()
        generated_ids.append(next_token_id)
    generated_tokens = [
        id_to_word[token_id]
        for token_id in generated_ids
    ]
    return " ".join(generated_tokens)
print("\nGENERATED TEXT")
prompts = [
    "artificial intelligence",
    "machine learning",
    "transformers",
    "python",
    "large language models"
]
for prompt in prompts:
    generated = generate_text(
        prompt,
        max_new_tokens=15
    )
    print("\nPrompt:",prompt)
    print("\nGenerated:",generated)
print("\nDAY 3 COMPLETE")