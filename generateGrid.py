import numpy as np
import pandas as pd

x = np.linspace(-20, 20, 160)
y= np.linspace(-20, 20, 160)


touple_xy_list = []
for i in x:
    for j in y:
        touple_xy = (i, j)
        touple_xy_list.append(touple_xy)
df = pd.DataFrame(touple_xy_list, columns=['emb_x', 'emb_y'])
print(df.head)
pd.DataFrame(df).to_csv("/home/bvadhera/huber/gridpoints.csv")