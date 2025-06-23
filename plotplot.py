import numpy as np
import matplotlib.pyplot as plt

grid = [ 
        [ [1, 5] , [-3, 0] , [2, 4], [-3, 1] ],
        [ [2, 2] , [-1, -2] , [0, 1], [0, 0] ],  
        [ [3, 1] , [4, 2] , [2, 1], [4, 2] ],
]

grid = np.array(grid)

u = grid[:,:,0]    # slice x direction component into u array
v = grid[:,:,1]    # slice y direction component into v array

plt.quiver(u, v)   # plot it!
plt.show()
