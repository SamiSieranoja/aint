import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
# https://www.kaggle.com/datasets/volodymyrgavrysh/heart-disease
heart_data = pd.read_csv('data/heart_disease.csv', delimiter=',')

# Split into features and target
X = heart_data.iloc[:, 0:13]
y = heart_data['target']

# Normalize features
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=101)

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Make it (N,1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define dataset and dataloader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the Neural Network
class HeartNN(nn.Module):
	def __init__(self):
		super(HeartNN, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(13, 40),     # 13 input features
			nn.ReLU(),
			nn.Linear(40, 40),
			nn.ReLU(),
			nn.Linear(40, 1),       # Output layer
			nn.Sigmoid()            # Because it's binary classification
		)

	def forward(self, x):
		return self.net(x)

# Instantiate model, define loss and optimizer
model = HeartNN()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 250
train_losses=[]
val_losses=[]
for epoch in range(num_epochs):
	model.train()
	train_losstmp=[]
	val_losstmp=[]
	for batch_X, batch_y in train_loader:
		optimizer.zero_grad()
		outputs = model(batch_X)
		loss = criterion(outputs, batch_y)

		train_losstmp+=[loss.item()]
		loss.backward()
		optimizer.step()
	# if (epoch+1) % 1 == 0:
	if True:
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
		model.eval()
		with torch.no_grad():
			y_pred_probs = model(x_test)
			loss = criterion(y_pred_probs, y_test)
			y_pred = (y_pred_probs > 0.5).float()
			# val_losses+=[loss.mean.item()]
			# val_losstmp+=[loss.item()]
			val_losses.append(loss.item())
		acc = accuracy_score(y_test.numpy(), y_pred.numpy())
		print(f"Accuracy: {acc:.4f}")
	train_losses+= [np.mean(train_losstmp)]
	

# Evaluation
model.eval()
with torch.no_grad():
	y_pred_probs = model(x_test)
	y_pred = (y_pred_probs > 0.5).float()

# Accuracy
acc = accuracy_score(y_test.numpy(), y_pred.numpy())
print(f"Accuracy: {acc:.4f}")


# Plotting
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# # Confusion Matrix
# cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
# disp.plot()
# plt.show()

