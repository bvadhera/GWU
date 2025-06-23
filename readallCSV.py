import pandas as pd
import os
import glob
  
  
# use glob to get all the csv files 
# in the folder
path  = "/home/bvadhera/huber/rl_results"
csv_files = glob.glob(os.path.join(path, "trained_embeddings*.csv"))
start_file = "/home/bvadhera/huber/rl_results/trained_embeddings4888.csv"
st_file = "/home/bvadhera/huber/rl_results/trained_embeddings4889.csv"
master_df = pd.read_csv(start_file) 
count = 4889
print("Started  Count = " + str(count))
# loop over the list of csv files
for f in csv_files:  
    # read the csv file
    if f == st_file or count > 4889 :
        df = pd.read_csv(f)
        master_df = pd.concat([master_df,df])
        count = count + 1
        if count == 5500:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count = count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)
        if count ==6000:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file) 
        if count ==6500:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)     
        if count ==7000:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)  
        if count == 7500:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)
        if count == 8000:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            print(str(count) + "- done")
            file_all_embeddings.flush()
            file_all_embeddings.close()
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)
        if count == 8500:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)         
        if count == 9000:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close()
            print(str(count) + "- done")
            count =count + 1
            start_file = "/home/bvadhera/huber/rl_results/trained_embeddings" +str(count) +".csv"
            master_df = pd.read_csv(start_file)     
        if count == 9999:
            file_all_embeddings = open('/home/bvadhera/huber/critic_stuff/all_embeddings_dump_' +str(count) +".csv",'w')
            master_df.to_csv(file_all_embeddings,index=False) 
            file_all_embeddings.flush()
            file_all_embeddings.close() 
            print(str(count) + "- done")

                  
  
