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

show_validation = True
show_loss = True

# Load dataset
# https://www.kaggle.com/datasets/volodymyrgavrysh/heart-disease
heart_data = pd.read_csv('data/heart_disease.csv', delimiter=',')

# Split into features and target
X = heart_data.iloc[:, 0:13]
y = heart_data['target'] #  0 = no disease and 1 = disease

# Normalize features
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Make it (N,1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define dataset and dataloader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_batch_size=1
# train_batch_size=32
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
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
			nn.Sigmoid()            # [0, 1] [0, inf] Because it's binary classification
		)

	def forward(self, x):
		return self.net(x)

# Instantiate model, define loss and optimizer
model = HeartNN()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop
num_epochs = 375
train_losses=[]
val_losses=[]
train_accuracies=[]
test_accuracies=[]
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
	if True:
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
		model.eval()
		with torch.no_grad():
			y_train_pred = (model(x_train) > 0.5).float()
			train_acc = accuracy_score(y_train.numpy(), y_train_pred.numpy())
			train_accuracies.append(train_acc)

			y_pred_probs = model(x_test)
			loss = criterion(y_pred_probs, y_test)
			y_pred = (y_pred_probs > 0.5).float()
			val_losses.append(loss.item())
			test_acc = accuracy_score(y_test.numpy(), y_pred.numpy())
			test_accuracies.append(test_acc)
		print(f"Train Accuracy: {train_acc:.4f}  Test Accuracy: {test_acc:.4f}")
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
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(train_accuracies, label='Train Accuracy')
if show_validation:
	ax1.plot(test_accuracies, label='Test Accuracy', color='green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy and Loss over epochs')

if show_loss:
	ax2 = ax1.twinx()
	ax2.plot(train_losses, label='Train Loss', color='orange', linestyle='--')
	if show_validation:
		ax2.plot(val_losses, label='Val Loss', color='red', linestyle='--')
	ax2.set_ylabel('Loss')
	lines2, labels2 = ax2.get_legend_handles_labels()
else:
	lines2, labels2 = [], []

lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()
plt.show()
# Confusion Matrix
# cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
# disp.plot()
# plt.show()

