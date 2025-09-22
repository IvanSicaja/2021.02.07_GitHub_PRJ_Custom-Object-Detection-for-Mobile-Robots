import numpy as np

b = np.arange(56).reshape(7,8)                         # Create array with values from 0 to 55 and reshape it to 7 rows and 8 columns
print("b array at start: \n",b)
print('___________________________' ) # Get first row

print(b[1:3,0:2])