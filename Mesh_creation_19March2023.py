import numpy as np
import matplotlib.pyplot as plt

# Mesh Parameters
NoEx = 10
NoEy = 50
WidthMesh = 10
HeightMesh = 50

# Derived Parameters
NoNx = NoEx+1
NoNy = NoEy+1
TNoN = NoNx*NoNy
TNoE = NoEx*NoEy
EsizeX = WidthMesh/NoEx
EsizeY = HeightMesh/NoEy
nodes = []
conn = []

for y in np.linspace(0.0, HeightMesh, NoNx):
    for x in np.linspace(0.0, WidthMesh, NoNy):
        nodes.append([x, y])
nodes = np.array(nodes)
#np.savetxt('nodes.txt',nodes)


for j in range(NoEy):
    for i in range(NoEx):
        n0 = i + j*NoNx
        conn.append([n0, n0 + 1, n0 + 1 + NoNx, n0 + NoNy])
conn = np.array(conn)
#np.savetxt('conn.txt',conn)

plt.plot(nodes[:,0],nodes[:,1], 'o')
for i, txt in enumerate(nodes):
        plt.annotate(txt, (nodes[i, 0], nodes[i, 1]), fontsize=4, color='red')
