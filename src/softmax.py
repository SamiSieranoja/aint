import torch
import torch.nn.functional as F


# Assuming we would get these values from the Neural Network:
vec = torch.tensor([-17, -17, 1.391, -17, -17, 2.643, -17, -17, 0.697, -17])


x = F.softmax(vec)
print([f"{v:.1f}" for v in x])
# After softmax:
# ['0.0', '0.0', '0.2', '0.0', '0.0', '0.7', '0.0', '0.0', '0.1', '0.0']


# Compare to ground truth:
ground_truth = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float)

# Cost function:
mse = torch.mean((x - ground_truth) ** 2)
print(f"MSE: {mse:.4f}")
# MSE: 0.0140


