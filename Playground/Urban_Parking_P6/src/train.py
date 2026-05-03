import torch
import torch.nn.functional as F
from model import GCN
from prepare_data import X, y, edge_index

# -----------------------------
# 1. Train/Test Split
# -----------------------------
num_nodes = X.shape[0]
train_size = int(0.8 * num_nodes)
torch.manual_seed(42)
indices = torch.randperm(num_nodes)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# -----------------------------
# 2. Model Setup
# -----------------------------
model = GCN(X.shape[1], 16, 2)  # increased hidden layer

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 3. Class Weights (IMPORTANT FIX)
# -----------------------------
class_counts = torch.bincount(y)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 80

for epoch in range(epochs):
    model.train()

    out = model(X, edge_index)

    loss = F.cross_entropy(out[train_idx], y[train_idx], weight=class_weights)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)

    train_acc = (pred[train_idx] == y[train_idx]).float().mean().item()
    test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# -----------------------------
# 5. Final Output
# -----------------------------
print("\nFinal Results:")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

print("\nSample Predictions:", pred[:10])
print("Actual Labels:", y[:10])

torch.save(model.state_dict(), "model.pth")