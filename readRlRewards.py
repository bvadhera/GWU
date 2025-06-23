import os


def read():
    print (os.listdir('/home/bvadhera/huber'))
    if os.path.isfile('/home/bvadhera/huber/rewards.txt'):
        print("File exists")
    # reading file
    file1_w = open('/home/bvadhera/huber/tesRLTD3-TotalReward.csv', 'w')
    file1_r = open('/home/bvadhera/huber/rewards.txt', 'r')
    while True:
        line = file1_r.readline()
        if not line:
            break
        if "total " in line: 
            file1_w.write(line)
            #line = file1_r.readline()
            #if "actionArray" in line:  
            #    print(" ")
            #else:
                #file1_w.write(line)
    file1_r.close()
    file1_w.close()
read()