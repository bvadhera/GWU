import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

df_xyx1y1 = pd.read_csv("/home/bvadhera/huber/gridpoints_DrivenActor.csv")
soa = df_xyx1y1.to_numpy()
X, Y, U, V = zip(*soa)

'''
plt.figure()
ax = plt.gca()
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
plt.draw()
plt.show()
'''
X = np.asarray(X)
Y= np.asarray(X)
U = np.asarray(U)
V = np.asarray(V)
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V,units='xy' ,scale=1)

plt.grid()

ax.set_aspect('equal')

plt.xlim(-20,20)
plt.ylim(-20,20)

plt.title('Plotting a vector',fontsize=10)

plt.savefig('plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')
#plt.show()
plt.close() 
