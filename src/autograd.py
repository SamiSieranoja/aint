#!/usr/bin/env python
import torch
# Example from https://arxiv.org/abs/2402.16020v2 calculated with pytorch 

x1 = torch.tensor([2.0], requires_grad=True)
x2 = torch.tensor([5.0], requires_grad=True)

v1 = torch.log(x1);
v2 = x1*x2 
v3 = torch.sin(x2) 
v4 = v1 + v2
v5 = v4 - v3

print("v1=", v1.item())
print("v2=", v2.item())
print("v3=", v3.item())
print("v4=", v4.item())
print("v5=", v5.item())

# Without this, grad is only accessible for x1,x2
for zz in [v1,v2,v3,v4,v5]: zz.retain_grad()

# Perform backpropagation
loss = v5
loss.backward()

print("Loss:", loss.item())

# Access the gradients
print("x'1=", x1.grad.item())
print("x'2=", x2.grad.item())
print("v'1=", v1.grad.item())
print("v'2=", v2.grad.item())
print("v'3=", v3.grad.item())
print("v'4=", v4.grad.item())
print("v'5=", v5.grad.item())

