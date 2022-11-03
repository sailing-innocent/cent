import numpy as np

items = np.ones([5,3])
for i in range(5):
    for j in range(3):
        items[i][j] = i + j
        
print(items)

count = items.shape[0]
n = items.shape[1]
sumlist = np.zeros(n)

for i in range(count):
    item = items[i,:] 
    for j in range(n):
        sumlist[j] = sumlist[j] + item[j] 

print(sumlist)

for i in range(count):
    item = items[i,:]
    for j in range(n):
        item[j] = item[j]/sumlist[j] 

print(items)