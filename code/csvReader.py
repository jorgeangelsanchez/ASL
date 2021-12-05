import csv
import numpy as np

def readCSV(filepath):
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader))
    #classif= np.zeros((data.shape[0],2))
    classif = {}
    labels = headers[1:]
    print(data)
    for i in range(data.shape[0]):
        for j in range(1,len(labels)):
            print(data[i][j])
            if data[i][j]!=0:
                print("TRUEEEE")
                classif[data[i][0]] = labels[j]
                

    

    print(labels)
    print("Type",type(labels))
    print()
    print(classif)

readCSV('C:/Users/tmcar/cs1430/ASL/data/test/_classes.csv')