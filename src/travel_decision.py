import torch
import torch.nn as nn
import torch.nn.functional as F

# Luennon Neuroverkko-esimerkin toteutus
class TravelDecision(nn.Module):
	def __init__(self):
		super(TravelDecision, self).__init__()
		self.fc1 = nn.Linear(2, 2)
		self.fc2 = nn.Linear(2, 1)
		self.fc3 = nn.Linear(1, 2)
		
	def forward(self, x):
		x = F.relu(self.fc1(x))
		print(x)
		x = self.fc2(x)
		print(x)
		x = F.relu(self.fc3(x))
		return x
		
model = TravelDecision()

# Disabloidaan automaattinen derivointi
# Asetetaan verkon painot
with torch.no_grad():
	# ensimmäinen rivi [-8,-1] vastaa h3 painoja
	# toinen rivi h4 painoja
	model.fc1.weight.copy_(torch.tensor([[-8,-1], [-1,-0.01]]))       
	model.fc1.bias.copy_(torch.tensor([6, 5]))
	
	model.fc2.weight.copy_(torch.tensor([[13,-1]]))       
	model.fc2.bias.copy_(torch.tensor([0]))
	
	model.fc3.weight.copy_(torch.tensor([[1],[-1]]))       
	model.fc3.bias.copy_(torch.tensor([0]))


print("Hyvä sää (0.2):")
print(model(torch.tensor([0.2,4.0])))

print("Huono sää (0.8):")
print(model(torch.tensor([0.8,4.0])))

print("Pitkä matka (440):")
print(model(torch.tensor([0.8,440.0])))




