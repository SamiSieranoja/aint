import torch
import torch.nn.functional as F

# First layer of neural network
A = torch.tensor([[1, 4], [0, -3]]) # Layer weights
bias = torch.tensor([-2, 1]) 
x = torch.tensor([2, 3]) # input vector
y = F.relu(A@x + bias)
print(y) # tensor([12,  0])

# Second layer of neural network
x = y # Using output from previous layer as input to next
A = torch.tensor([[-2, 7], [3, 9]]) # Layer weights
bias = torch.tensor([3, 4]) 
y = F.relu(A@x + bias)
print(y) # tensor([ 0, 40])
