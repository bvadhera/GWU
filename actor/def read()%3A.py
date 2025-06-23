import os


def read():
    print (os.listdir())
    # reading file
    file1_w = open('tesRLTD3.csv', 'w')
    file1_r = open('tesRLTD3.txt', 'r')
    while True:
        line = file1_r.readline()
        if not line:
            break
        if "reward " in line: 
            line = file1_r.readline()
            print(line)
            file1_w.write(line)
    file1_r.close()
    file1_w.close()
read()