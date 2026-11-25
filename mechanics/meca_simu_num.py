import numpy as np
import matplotlib.pyplot as plt


g = 9.81      
l = 0.1     
m = 1.0     
theta0 = np.pi / 6
v0 = 1.0 

dt = 0.001     
t_max = 10    
t = np.arange(0, t_max, dt)

lambdas = [2, 20, 50]  
colors = ['red', 'green', 'blue'] 

def euler2(theta0, v0, lambd, t, m, l, g):
    """Méthode d'Euler """
    n = len(t)
    theta = np.zeros(n)
    omega = np.zeros(n)
    
    theta[0] = theta0
    omega[0] = v0

    for i in range(n - 1):
        domega = -(lambd/m)*omega[i] - (g/l)*np.sin(theta[i])
        omega[i+1] = omega[i] + dt * domega
        theta[i+1] = theta[i] + dt * omega[i]
        
    return theta, omega

plt.style.use('dark_background')
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

for i, (lambd, color) in enumerate(zip(lambdas, colors)):
    theta, _ = euler2(theta0, v0, lambd, t, m, l, g)
    axs[i].plot(t, theta, label=f"λ = {lambd} kg/s", color=color)
    axs[i].set_ylabel("θ (rad)", color='white')
    axs[i].set_title(f"Évolution de θ(t) - λ = {lambd} kg/s", color='white')
    axs[i].grid(True, color='gray')
    axs[i].legend(facecolor='black', edgecolor='white', labelcolor='white')
    axs[i].tick_params(colors='white')

axs[-1].set_xlabel("Temps (s)", color='white')

plt.tight_layout()
plt.show()