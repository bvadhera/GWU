import pandas as pd
import csv

input_DataSet = "/home/bvadhera/huber/rl_results/trained_embeddings.csv"
df = pd.read_csv(input_DataSet)
print("df - Display number of rows, columns, etc.: ")
print(df.info())
print("df - Get the number of rows:") 
print(len(df))
print("df - Get the number of columns: ")
print(len(df.columns))
print("df - Get the number of rows and columns:")
print(df.shape)
print("df - Get the number of elements: df.size")


df_uniq = df.drop_duplicates()
print("df_uniq - Display number of rows, columns, etc.: ")
print(df_uniq.info())
print("df_uniq - Get the number of rows:") 
print(len(df_uniq))
print("df_uniq - Get the number of columns: ")
print(len(df_uniq.columns))
print("df_uniq - Get the number of rows and columns:")
print(df_uniq.shape)
print("df_uniq - Get the number of elements: df.size")