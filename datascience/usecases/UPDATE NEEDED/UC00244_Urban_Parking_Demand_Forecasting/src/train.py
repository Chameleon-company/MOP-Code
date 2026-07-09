import torch
import torch.nn.functional as F
from model import GCN
from prepare_data import X, y, edge_index


# 1. Train/Test Split
num_nodes = X.shape[0]
train_size = int(0.8 * num_nodes)

torch.manual_seed(42)
indices = torch.randperm(num_nodes)
train_idx = indices[:train_size]
test_idx = indices[train_size:]


# 2. Model Setup
model = GCN(X.shape[1], 16, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 3. Class Weights
class_counts = torch.bincount(y).float()

# avoid division by zero
class_counts[class_counts == 0] = 1

total = class_counts.sum()

# Strong balancing
class_weights = total / (2 * class_counts)
class_weights = class_weights.to(X.device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)


# 4. Training Loop
epochs = 80
best_test_acc = 0
best_model_state = None

for epoch in range(epochs):
    model.train()

    out = model(X, edge_index)

    loss = F.cross_entropy(out[train_idx], y[train_idx], weight=class_weights)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    model.eval()
with torch.no_grad():
    out_eval = model(X, edge_index)

    # Convert logits → probabilities
    probs = torch.softmax(out_eval, dim=1)

    # Apply custom threshold
    threshold = 0.4
    pred = (probs[:, 1] > threshold).long()

    train_acc = (pred[train_idx] == y[train_idx]).float().mean().item()
    test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()

    print("\nThreshold used:", threshold)
    print("Sample predictions:", pred[:10])

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = model.state_dict()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# 5. Save best model
if best_model_state:
    torch.save(best_model_state, "model.pth")


# 6. Final Output
print("\nFinal Results:")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Best Test Accuracy:", best_test_acc)

print("\nSample Predictions:", pred[:10])
print("Actual Labels:", y[:10])


# 7. Debug: Probabilities
probs = torch.softmax(out_eval, dim=1)
print("\nPrediction probabilities (first 5):")
print(probs[:5])
print("\nPredicted classes (first 10):")
print(probs.argmax(dim=1)[:10])