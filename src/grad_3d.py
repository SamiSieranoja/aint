import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# Define the function to minimize
def f(x, y):
    return x**2 + 2*y**2 

# Compute the gradient of the function
def grad_f(x, y):
    df_dx = 2 * x
    df_dy = 4 * y
    return np.array([df_dx, df_dy])

# Gradient descent parameters
alpha = 0.1  # Learning rate
iterations = 50  # Number of iterations
x_vals, y_vals = [2], [2]  # Starting point

# Perform gradient descent
for i in range(iterations):
    x = x_vals[-1]
    y = y_vals[-1]
    grad = grad_f(x, y)
    fval = f(x,y)
    
    magnitude = math.sqrt(x*x + y*y)
    print(f"{i},{round(x_vals[-1],3)},{round(y_vals[-1],3)},{[round(grad[0],3),round(grad[1],3)]},{round(magnitude,2)},{fval}")
    
    new_x = x_vals[-1] - alpha * grad[0]
    new_y = y_vals[-1] - alpha * grad[1]
    x_vals.append(new_x)
    y_vals.append(new_y)

# Generate data for the contour plot
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot the contour and gradient descent path
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X, Y, Z, levels=20, cmap='jet')
ax.plot(x_vals, y_vals, 'ro-', markersize=4, label='Gradient Descent Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Gradient Descent Visualization')
plt.show()

# 3D visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.6)
ax.plot(x_vals, y_vals, f(np.array(x_vals), np.array(y_vals)), 'ro-', markersize=4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Gradient Descent in 3D')
plt.show()

