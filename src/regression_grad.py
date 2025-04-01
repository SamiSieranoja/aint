import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

random.seed(8888)

# Example based on:
# https://github.com/chasinginfinity/ml-from-scratch/blob/master/02%20Linear%20Regression%20using%20Gradient%20Descent/.ipynb_checkpoints/Linear%20Regression%20using%20Gradient%20Descent-checkpoint.ipynb

ds = np.genfromtxt('data/ice_cream_linear.txt')

# Data:
X = ds[:,0]
Y = ds[:,1]

# Model parameters:
m = 0.1
c = 20

# Learning rates for m and c:
lr_m = 0.100
lr_c = 0.1

epochs = 3000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

Y_pred = m*X + c

grad_lines=[]
for i in range(epochs): 
	Y_pred = m*X + c  # The current predicted value of Y
	D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
	D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
	m = m - lr_m * D_m  # Update m
	c = c - lr_c * D_c  # Update c

	loss = sum((Y - Y_pred)**2)
	
	if np.isnan(loss): 
		break
	
	
	gradline=[[min(X), max(X)], [min(Y_pred), max(Y_pred)],m,c,loss]
	grad_lines.append(gradline)
	
	print(f"loss={loss} m={m} c={c}")

Y_pred = m*X + c
fig, ax = plt.subplots()
plt.scatter(X, Y)

t1 = ax.text(0.05, 0.95, 'Upper Left Text', fontsize=12, color='black',
         horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
         
plt.xlabel("Temperature (C)", fontsize=14)
plt.ylabel("Icre Cream sales (litres)", fontsize=14)
       
paused = False
def on_key(event):
	global paused
	if event.key == ' ':  # Space bar to pause/resume
		paused = not paused
fig.canvas.mpl_connect('key_press_event', on_key)

gl = grad_lines[0]
line, = ax.plot(gl[0], gl[1], color='red') # predicted

def animate(i):
	if not paused:
		gl = grad_lines[i]
		line.set_data(gl[0], gl[1]) # predicted
		m = round(gl[2],3)
		c = round(gl[3],3)
		loss = round(gl[4],2)
		t1.set_text(f"m={m} c={c}\nloss={loss}\ny=m*x+c")
	return line, t1

interval=1e3 #slow
interval=1e2 #fast
ani = animation.FuncAnimation(fig, animate, frames=len(grad_lines), 
			interval=interval, blit=True)

plt.show()

