import torch
import torch.nn as nn
import torch.optim as optim

data = [
    ("I love this movie", 1),
    ("this film was great", 1),
    ("absolutely fantastic work", 1),
    ("I hated this movie", 0),
    ("this film was terrible", 0),
    ("absolutely worst experience", 0)
]

vocab = set()
for text, _ in data:
    for word in text.split():
        vocab.add(word)

# Create a word-to-index mapping dictionary
# Index 0 is reserved for "<PAD>" to ensure sentences are of equal length
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx["<PAD>"] = 0
vocab_size = len(word_to_idx)

# Find the maximum sentence length to standardise dimensions
max_len = max(len(text.split()) for text, _ in data)

# Convert raw text tokens into padded numerical arrays
inputs = []
labels = []

for text, label in data:
    words = text.split()
    # Replace words with indices; pad the rest of the array with 0s (<PAD>)
    indexed = [word_to_idx[w] for w in words] + [0] * (max_len - len(words))
    inputs.append(indexed)
    labels.append(label)

# Convert Python lists into PyTorch multi-dimensional Tensors
X = torch.tensor(inputs, dtype=torch.long)
Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Shape: (6, 1)


class SimpleTextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleTextRNN, self).__init__()

        # Layer 1: Transforms sparse integer IDs into dense semantic vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Layer 2: Main recurrent engine.
        # batch_first=True aligns layout format to (Batch, Sequence Length, Feature Dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Layer 3: Linear layer mapping final memory state to a single score output
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        # text layout: [batch_size, seq_len]
        embedded = self.embedding(text)  # output layout: [batch_size, seq_len, embedding_dim]

        # Process structural stream.
        # 'rnn_out' records hidden steps over time; 'hidden' records final chronological summary
        rnn_out, hidden = self.rnn(embedded)

        # Pull the final hidden state summary tensor
        last_hidden = hidden.squeeze(0)  # changes format from [1, batch_size, hidden_dim] to [batch_size, hidden_dim]

        # Compute final activation logit maps
        output = self.fc(last_hidden)
        return torch.sigmoid(output)     # Squashes values safely between 0 and 1


# Hyperparameters Configurations
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
MODEL = SimpleTextRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

# Loss Formulation & Optimization Strategies
criterion = nn.BCELoss()
optimizer = optim.Adam(MODEL.parameters(), lr=0.01)


print("--- Launching RNN Training Optimization ---")
for epoch in range(100):
    optimizer.zero_grad()             # Reset standard gradient parameter accumulation
    predictions = MODEL(X)             # Execute forward matrix pass
    loss = criterion(predictions, Y)   # Evaluate overall error bounds
    loss.backward()                    # Compute Backpropagation derivatives
    optimizer.step()                   # Update parameters weights

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d}/100 | Training Loss Matrix Error: {loss.item():.4f}")


print("\n--- Running Evaluation Inference Check ---")
MODEL.eval()
with torch.no_grad():
    test_sentence = "I love fantastic film"

    # Process the text token sequence matching preparation standards
    test_words = test_sentence.split()

    # Graceful fallback: use 0 if a token isn't in training vocabulary
    test_indexed = [word_to_idx.get(w, 0) for w in test_words] + [0] * (max_len - len(test_words))

    test_tensor = torch.tensor([test_indexed], dtype=torch.long)
    prediction = MODEL(test_tensor).item()

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Input Text String: {test_sentence}")
    print(f"Raw Sigmoid Output Score: {prediction:.4f}")
    print(f"Determined Sentiment Classification: {sentiment}")




    
        
    