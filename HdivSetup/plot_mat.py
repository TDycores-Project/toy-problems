import re
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_mat.py mat.dat")
    sys.exit(1)
    
N = None
for line in open(sys.argv[1]):
    match = re.search("row\s(\d+):",line)
    if match:
        N = int(match.groups(1)[0])
N += 1

x = []
y = []
c = []
for line in open(sys.argv[1]):
    match = re.search("row\s(\d+):",line)
    if match:
        row = int(match.groups(1)[0])
        cols = re.findall("\s\((\d+),\s(\d*\.\d*)\)",line)
        for col in cols:
            x.append(row)
            y.append(  int(col[0]))
            c.append(float(col[1]))
            
fig = plt.figure(figsize=(8+8*0.2,8),tight_layout=True)
ax = plt.scatter(x,N-np.asarray(y),c=c)
plt.colorbar(ax)
plt.show()
