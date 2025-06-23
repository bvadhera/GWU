import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

listx = []
listy = []
listz = []
listacc = []

for idx in range(50):
  # 50 linearly spaced numbers
  start_point = 1
  x = random.randint(start_point,2)
  listx.append(x)
  y = random.randint(start_point,23)
  listy.append(y)
  if x == 1:
    z = 0
  else:
    z = random.randint(start_point,23)
  listz.append(z)
  accuracy = -((x-0)**2 + (y-20)**2  + (z-15)**2 ) 
  listacc.append(accuracy) 

print (listx)
print ("/n")
print (listy)
print ("/n")
print (listz)
print ("/n")
print (listacc)
print ("/n")

listacc = [float(x) for x in listacc] 
print (listacc)
print ("max_value" + "/n")
max_value = max(listacc)
min_value = min(listacc)
dividing_factor = max_value - min_value
print (dividing_factor)
print ("/n")
listacc[:] = [number - min_value for number in listacc]
print (listacc)
print ("/n")
# Normalize to have values between 0 - 1
listacc_normalized = [x / dividing_factor for x in listacc]

print (listacc_normalized)

a_zip = zip(listx, listy, listz,  listacc_normalized)
 
zipped_list = list(a_zip)
 
print(zipped_list)
pd.DataFrame(zipped_list).to_csv("/home/bvadhera/huber/forthNetwork.csv")