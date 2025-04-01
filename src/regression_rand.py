import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Estimation of linear regression parameters using random search 

random.seed(8888)

# Example based on:
# https://github.com/chasinginfinity/ml-from-scratch/blob/master/02%20Linear%20Regression%20using%20Gradient%20Descent/.ipynb_checkpoints/Linear%20Regression%20using%20Gradient%20Descent-checkpoint.ipynb

ds = np.genfromtxt('data/ice_cream_linear.txt')
X = ds[:,0]
Y = ds[:,1]

# Initial model parameters:
m = 0.1
c = 20

# Learning rates for m and c:
lr_m = 0.1
lr_c = 0.5

epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

Y_pred = m*X + c
loss_best = loss = sum((Y - Y_pred)**2)

# Performing Random Search 
grad_lines=[]
for i in range(epochs): 
	rand1 = random.uniform(-1.0, 1.0)
	rand2 = random.uniform(-1.0, 1.0)
	m_new = m + rand1*lr_m
	c_new = c + rand2*lr_c
	Y_pred = m_new*X + c_new  # The current predicted value of Y
	loss = sum((Y - Y_pred)**2)
	if loss < loss_best:
		m = m_new		
		c = c_new		
		loss_best = loss
		
	Y_pred_best = m*X + c  # The best found predicton
	
	gradline=[[min(X), max(X)], [min(Y_pred), max(Y_pred)],[min(Y_pred_best), max(Y_pred_best)],m,c,loss]
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
line2, = ax.plot(gl[0], gl[2], color='green') # predicted

def animate(i):
	if not paused:
		gl = grad_lines[i]
		line.set_data(gl[0], gl[1]) # predicted
		line2.set_data(gl[0], gl[2]) # predicted
		m = round(gl[3],3)
		c = round(gl[4],3)
		loss = round(gl[5],2)
		t1.set_text(f"m={m} c={c}\nloss={loss}\ny=m*x+c")
	return line, line2, t1

interval=1e3 #slow
interval=1e2 #fast
ani = animation.FuncAnimation(fig, animate, frames=len(grad_lines), 
			interval=interval, blit=True)

plt.show()

