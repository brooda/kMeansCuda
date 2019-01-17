import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


lenOfInput = int(sys.argv[1])
k = int(sys.argv[2])

k_centers = [(2, 2, 2), (6, 6, 6), (6, 6, 0), (2,2,0), (2, 6, 2)]

xs = []
ys = []
zs = []
classes = []

for i in range(k):
    centre = k_centers[i]
    xc = centre[0]
    yc = centre[1]
    zc = centre[2]
    
    
    for j in range(lenOfInput):
        deltax = np.random.normal()
        deltay = np.random.normal()
        deltaz = np.random.normal()

        xs.append(xc + deltax)
        ys.append(yc + deltay)
        zs.append(zc + deltaz)
        
xs = np.round(xs, 2)
ys = np.round(ys, 2)
zs = np.round(zs, 2)

df = pd.DataFrame({"x" : xs, "y" : ys, "z" : zs})

df.to_csv(f"{k}/{k*lenOfInput}.csv", index=False, header=False)
