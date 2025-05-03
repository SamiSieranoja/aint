import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

import re
import pandas as pd
df = pd.read_csv('data/animal_ds.txt',header=None)

def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

corpus = list(df.iloc[:, 0])
tokenized = [re.findall(r'\b\w+\b', l.lower()) for l in corpus]
words = {}
for l in tokenized:
	for w in l:
		words[w]=True

vocab = words
words["<EOS>"]=True
words["<SOS>"]=True
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
num_words = len(vocab)
print(f"num_words:{num_words}, device={device}")

emb_dim=4
class NextWordPredictor(nn.Module):
	def __init__(self, input_dim, hidden_dim, vocab_size):
		super(NextWordPredictor, self).__init__()
		
		# Embedding layer
		self.embed = nn.Linear(num_words, emb_dim)
		
		# One hidden layer 
		self.fc1 = nn.Linear(emb_dim*3, vocab_size)
		
	def forward(self, x):
		vec1 = x[0]
		vec2 = x[1]
		vec3 = x[2]
		
		#One-hot vektorista embedding vektori
		emb1 = F.leaky_relu(self.embed(vec1))
		emb2 = F.leaky_relu(self.embed(vec2))
		emb3 = F.leaky_relu(self.embed(vec3))
		
		# Yhdistetään vektorit (concatenate)
		contextvec=torch.cat((emb1,emb2,emb3),dim=1)
		
		x = F.leaky_relu(self.fc1(contextvec))
		return x


def create_ds():
	X = []
	y = []
	v1s=[] # First context word vectors
	v2s=[] # Second context word vectors
	v3s=[]
	
	# Extract words
	tokenized = [re.findall(r'\b\w+\b', l.lower()) for l in corpus]

	for tokens in tokenized:
		tokens = ["<SOS>","<SOS>","<SOS>"] + tokens +  ["<EOS>"]
		for i in range(len(tokens) - 3):
			w1, w2, w3, w4 = tokens[i:i+4]
			if all(w in word_to_idx for w in [w1, w2, w3, w4]):
				v1=F.one_hot(torch.tensor(word_to_idx[w1]),num_words).float()
				v2=F.one_hot(torch.tensor(word_to_idx[w2]),num_words).float()
				v3=F.one_hot(torch.tensor(word_to_idx[w3]),num_words).float()
				v1s.append(v1.view(1,num_words))
				v2s.append(v2.view(1,num_words))
				v3s.append(v3.view(1,num_words))
				y.append(word_to_idx[w4])
	v1s=torch.cat(v1s, dim=0)
	v2s=torch.cat(v2s, dim=0)
	v3s=torch.cat(v3s, dim=0)
	y = torch.tensor(y, dtype=torch.long).to(device)
	return [[v1s,v2s,v3s],y]


input_dim = num_words
hidden_dim = 512
model = NextWordPredictor(input_dim, hidden_dim, num_words)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)
netfn="animal_weights.pth"
prev = False
epochs = 200
if os.path.isfile(netfn): 
	model.load_state_dict(torch.load(netfn, weights_only=True))
	prev = False
	epochs = 0

model.to(device)

(X,y) = create_ds()
# Training loop
for epoch in range(epochs):
	outputs = model(X)
	loss = criterion(outputs, y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if (epoch+1) % 2 == 0:
		print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model to file
torch.save(model.state_dict(), netfn)
		
# Predict context given words w1,w2, w3
def predict_next(w1, w2, w3,topk=0):
	if all(w in word_to_idx for w in [w1, w2, w3]):
	
		v1=to_one_hot(word_to_idx[w1],num_words).to(device).view(1,-1)
		v2=to_one_hot(word_to_idx[w2],num_words).to(device).view(1,-1)
		v3=to_one_hot(word_to_idx[w3],num_words).to(device).view(1,-1)
		with torch.no_grad():
			output = F.softmax(model([v1,v2,v3]),dim=1)
			predicted_idx = torch.multinomial(output, num_samples=1).item()
			if topk > 0:
				topk_values, topk_indices = torch.topk(output, topk)
			for i in range(0,topk):
				val = round(topk_values[0,i].item(),3)
				word = idx_to_word[topk_indices[0,i].item()]
				print(f"    word={word} probability={val}")
			return idx_to_word[predicted_idx]
	else:
		return "Unknown word(s)"

model.eval()
def generate_sentence(context = ["<SOS>", "<SOS>", "<SOS>"]):
	sentence = " ".join(context)
	for i in range(0,20):
		w = predict_next(context[0],context[1],context[2])
		context = context[1:]+[w]
		# print(w)
		# print(context)
		sentence += " " + w
		if w == "<EOS>":
			break
	print(re.sub("<[ES]OS>\\s*","",sentence))
	
# Seuraavat tekstit ei ole osa treeni-datajoukkoa:
# "mackarel is a fish", " "monkey is a mammal" , "pigeon is a bird"
# Neuroverkon pitäisi kuitenkin pystyä päättelemään nämä

context = "mackerel is a".split(" ")
print(" ".join(context))
predict_next(context[0],context[1],context[2],4)
print(" ")

context = "monkey is a".split(" ")
print(" ".join(context))
predict_next(context[0],context[1],context[2],4)
print(" ")

context = "pigeon is a".split(" ")
print(" ".join(context))
predict_next(context[0],context[1],context[2],4)
print(" ")

print("Random sentences:")
for i in range(0,10): generate_sentence()


