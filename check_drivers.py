import pyodbc as dbc
import numpy as np
from tensorflow.python.client import device_lib

print('list of drivers\n')
for driver in dbc.drivers():
    print(driver)

list1 = []
list2 = [1,2,3,4]
for i in range(0,3):
    list1.append(list2)
ar = np.array(list1)
print('Arr list1:\n', ar)
print('Array column:\n', ar[1,:])

print('Tensorflow test:\n')
print(device_lib.list_local_devices())