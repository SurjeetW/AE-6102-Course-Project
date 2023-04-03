import numpy as np
import math
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

for y in np.linspace(0.0, HeightMesh, NoNy):
    for x in np.linspace(0.0, WidthMesh, NoNx):
        nodes.append([x, y])
nodes = np.array(nodes)
#np.savetxt('nodes.txt',nodes)


for j in range(NoEy):
    for i in range(NoEx):
        n0 = i + j*NoNx
        conn.append([n0, n0 + 1, n0 + 1 + NoNx, n0 + NoNx])
conn = np.array(conn)
#np.savetxt('conn.txt',conn)

plt.plot(nodes[:,0],nodes[:,1], 'o')
for i, txt in enumerate(nodes):
        plt.annotate(txt, (nodes[i, 0], nodes[i, 1]), fontsize=4, color='red')

########### material constitutive model - plane strain ##########
E = 100.0
v = 0.48
C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v,     v,     0.0],
								      [    v, 1.0-v,     0.0],
								      [  0.0,   0.0,   0.5-v]])
################# gradient of shape function: 4 noded lower order element ##########################################
def gradshape(xi):
	x,y = tuple(xi)
	dN = [[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],
		  [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]]
	return 0.25*np.array(dN)
############################### Element stiffness matrix and Global stiffness matrix creation #####
K = np.zeros((2*TNoN, 2*TNoN))
q4 = [[x/math.sqrt(3.0),y/math.sqrt(3.0)] for y in [-1.0,1.0] for x in [-1.0,1.0]]
B = np.zeros((3,8))
print(conn.shape) # Total no. of elements in model
for c in conn:
	xIe = nodes[c,:]
	Ke = np.zeros((8,8)) # Initialization of element stiffness matrix
	for q in q4:
		dN = gradshape(q)
		J  = np.dot(dN, xIe).T
		dN = np.dot(np.linalg.inv(J), dN)
		B[0,0::2] = dN[0,:]
		B[1,1::2] = dN[1,:]
		B[2,0::2] = dN[1,:]
		B[2,1::2] = dN[0,:]
		Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)
	for i,I in enumerate(c):
		for j,J in enumerate(c):
			K[2*I,2*J]     += Ke[2*i,2*j]
			K[2*I+1,2*J]   += Ke[2*i+1,2*j]
			K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
			K[2*I,2*J+1]   += Ke[2*i,2*j+1]
