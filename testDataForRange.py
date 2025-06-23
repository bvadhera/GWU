import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import itertools
import csv
from csv import DictReader

list_legal_x =[]
list_illlegal_x = []
list_legal_y =[]
list_illlegal_y = []
list_legal_z =[]
list_illlegal_z = []
list_illlegal_acc = []
list_legal_acc = []
illigal_value = False
illlegal_zipped_list = []
legal_zipped_list = []
combinedList =[]

# generate random flot value with step of 0.1
def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

def validateX(x,illigal_value):
  x_ = 0.0
  # Save  for equvalent value after roundup/down and illigal/legal 
  if (x >=0.6  and  x <= 1.4) or (x >=1.6  and  x <= 2.4): 
    if (x >=0.6  and  x <= 1.4): # Legal equivalent to 1.0
      x_ = 1.0
    elif (x >=1.6  and  x <= 2.4): # Legal equivalent to 2.0
      x_ = 2.0
    list_legal_x.append(x)
    illigal_value = False
  else: # illLegal 
    illigal_value = True
    x_ = x
    list_illlegal_x.append(x)
  return illigal_value,x_

def validateY(y,illigal_value):
  y_ = round(y)
  rounded_Diff =  round( abs(y - y_),2) 
   
  # Save as touple for equvalent value after roundup/down and illigal/legal 
  if (y > 23.4): # Illigal
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  elif (y < 0.6): # Illigal as first layer can not have nodes = 0
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  elif (rounded_Diff <= 0.4): # round down, legal
    list_legal_y.append(y)
    illigal_value = False
  else:
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  return illigal_value,y_

def validateZ(z,illigal_value, x_):
  z_ = round(z)
  rounded_Diff =   round( abs(z - z_),2) 
   
  # Save as touple for equvalent value after roundup/down and illigal/legal 
  if (z > 23.4): # Illigal
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif ( (z > 0.6)  and  (x_ == 1) ):  # Illigal as second layer can not have nodes if x_ = 1
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif ((z < 0.6) and  (x_ == 2) ):  # Illigal as second layer should have nodes if x_ = 2 
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif (rounded_Diff <= 0.4): # round down, legal
    list_legal_z.append(z)
    illigal_value = False
  else:
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  return illigal_value,z_

for idx in range(100000):
  # 50 linearly spaced numbers
  start_point = 0.0                                     
  x = randrange_float(start_point, 3.0, 0.1)  #(0 - 0.5) and (1.4-1.6) (2.5-3.0 ) accuracy`` 0 illigal
                                       #(0.6-1.4 mapped to 1 accuracy) (1.6-2.4 mapped to 2 accuracy)
  illigal_value_x,x_ = validateX(x,illigal_value) 
  y = randrange_float(start_point, 24.0, 0.1)
  illigal_value_y,y_  = validateY(y,illigal_value)
  z = randrange_float(start_point, 24.0, 0.1)
  illigal_value_z,z_  = validateZ(z,illigal_value,x_)

  # if illigal network then generate Min accuracy
  if ( illigal_value_x or  illigal_value_y or  illigal_value_z ):
    accuracy = -((2)**2 + (12)**2 + (15)**2)
    list_illlegal_acc.append(accuracy) 
    # Make a touple for all points here
    toupleIllLegal = (x,y,z,accuracy)
    illlegal_zipped_list.append(toupleIllLegal)
  else:
    accuracy = -((x_-2)**2 + (y_-12)**2  + (z_-8)**2 ) 
    list_legal_acc.append(accuracy) 
    toupleLegal = (x,y,z,accuracy)
    legal_zipped_list.append(toupleLegal)

#legal_zip = zip(list_legal_x, list_legal_y, list_legal_z,  list_legal_acc)
#legal_zipped_list = list(legal_zip)
#illLegal_zip = zip(list_illlegal_x, list_illlegal_y, list_illlegal_z,  list_illlegal_acc)
#illlegal_zipped_list = list(illLegal_zip)
 
# Now to get length of both legal and illigal list
sizeLegalList = len(list_legal_acc)
sizeillLegalList = len(list_illlegal_acc)
size_legal_zipped_list = len(legal_zipped_list)
size_illlegal_zipped_list = len(illlegal_zipped_list)


# Need two files and get 50 % of each network in combined file.
for combination in itertools.zip_longest(legal_zipped_list, illlegal_zipped_list):
  print(combination)
  combinedList.append(combination)
pd.DataFrame(combinedList).to_csv("/home/bvadhera/huber/combinedNetwork10000.csv")


# Create a cleaned file
writecleanedFile = "/home/bvadhera/huber/combinedNetwork10000Cleaned.csv"
mycleanedFile = open(writecleanedFile, 'w')
headercleaned = ['num_layers', 'num_node1','num_node2','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','test_acc'] 
writercleaned = csv.DictWriter(mycleanedFile, fieldnames=headercleaned) 
writercleaned.writeheader()

# Now read from this file and take first min length iterations and sort them into new csv file for testing
# Change the dividing factor by 24 and 3
inputNN_Architectures_DataSet = "/home/bvadhera/huber/combinedNetwork10000.csv"
df = pd.read_csv(inputNN_Architectures_DataSet)
print(df.head())
df = df.reset_index()  # make sure indexes pair with number of rows
for index, row in df.iterrows():
    print(row['0'], row['1'])
    if ( (row['0'] == row['0']) and (row['1'] == row['1']) ):  # only if both legal and illigal rows are avaiable we want 50-50 data mix
      touple1 = row['0'] 
      touple11 = touple1.replace("(", "")
      touple111 = touple11.replace(")", "")
      touple1111 = touple111.replace(" ", "")
      num_layers, num_node1, num_node2, test_acc = touple1111.split(',')
      writercleaned.writerow({ 
          'num_layers' : num_layers,
          'num_node1' : num_node1,
          'num_node2' : num_node2,
          'F1':0, 'F2':1,'F3':0, 'F4':0,'F5':0,
          'F6':0,'F7':0, 'F8':0, 'F9':0,'F10':0,
          'test_acc' : test_acc
          })
      touple2 = row['1'] 
      touple21 = touple2.replace("(", "")
      touple211 = touple21.replace(")", "")
      touple2111 = touple211.replace(" ", "")
      num_layers, num_node1, num_node2, test_acc = touple2111.split(',')
      writercleaned.writerow({ 
          'num_layers' : num_layers,
          'num_node1' : num_node1,
          'num_node2' : num_node2,
          'F1':0, 'F2':1,'F3':0, 'F4':0,'F5':0,
          'F6':0,'F7':0, 'F8':0, 'F9':0,'F10':0,
          'test_acc' : test_acc
          })
    else:
      print("one row is finished")
      break
mycleanedFile.close()
# Now read back the combinedNetwork10000Cleaned.csv
df = pd.read_csv("/home/bvadhera/huber/combinedNetwork10000Cleaned.csv")
print(df.head())

normalizeProp = list(df.columns.values)
print(normalizeProp)
accuracy_list = df['test_acc'].to_list()
print(len(accuracy_list))
max_value = max(accuracy_list)
min_value = min(accuracy_list)
# Find Max and Min num of layers
listx = df['num_layers'].to_list()
maxNumOfLayers  = max(listx)
minNumOfLayers  = min(listx) 

# Writing a data to file
fileMinMax = open('/home/bvadhera/huber/secondNetwork10000MinMax.csv', 'w')
 
fileMinMax.write(str(max_value) + ',')
fileMinMax.write(str(min_value) + ',')
fileMinMax.write(str(maxNumOfLayers) + ',')
fileMinMax.write(str(minNumOfLayers))  

  
# Closing file
fileMinMax.close()

dividing_factor = max_value - min_value
print (dividing_factor)
print ("/n")
accuracy_list[:] = [number - min_value for number in accuracy_list]
print (accuracy_list)
print ("/n")
# Normalize to have values between 0 - 1
accuracy_list_normalized = [x / dividing_factor for x in accuracy_list]
print (accuracy_list_normalized)

normalizeProp.remove('test_acc')
print("normalizeProp")
print(normalizeProp)
df_ = df[normalizeProp]
df_['test_acc'] = accuracy_list_normalized
print(df_.head())
prop = list(df_.columns.values)
print(prop)
df_.to_csv("/home/bvadhera/huber/secondNetwork" + str(len(accuracy_list)) +".csv")


