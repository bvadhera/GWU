import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

listx = []
listy = []
listz = []
listacc = []

for idx in range(1000):
  # 50 linearly spaced numbers
  start_point = 0
  x = random.randint(start_point,2)
  listx.append(x)
  y = random.randint(start_point,23)
  listy.append(y)
#  if x == 1:
#    z = 0
#  else:
  z = random.randint(start_point,23)
  listz.append(z)
  #if illigal network example  (0,1,2)(1,0,8)(2,1,0) (1,2,2) then generate Min accuracy
  if ( (x==0) or (x==1 and ((y==0) or (z !=0))) or  ((x==2) and ((y==0) or (z==0)))):
    accuracy = -((2)**2 + (12)**2 + (15)**2)   
  else:
    accuracy = -((x-2)**2 + (y-12)**2  + (z-8)**2 ) 
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

# Find Max and Min num of layers
maxNumOfLayers  = max(listx)
minNumOfLayers  = min(listx) 


fileMinMax = open('/home/bvadhera/huber/secondNetwork500MinMax.csv', 'w')


  
# Writing a data to file
fileMinMax.write(str(max_value) + ',')
fileMinMax.write(str(min_value) + ',')
fileMinMax.write(str(maxNumOfLayers) + ',')
fileMinMax.write(str(minNumOfLayers))  

  
# Closing file
fileMinMax.close()
  
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

pd.DataFrame(zipped_list).to_csv("/home/bvadhera/huber/secondNetwork500.csv")