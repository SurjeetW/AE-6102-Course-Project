import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt


def mesh_generate(HeightMesh, WidthMesh, NoNx, NoNy, NoEx, NoEy, output_dir):
    nodes = []
    conn = []
    pltn = 'FE'
    for y in np.arange(0.0, HeightMesh+1, 1):
        for x in np.arange(0.0, WidthMesh+1, 1):
            nodes.append([x, y])
    nodes = np.array(nodes).astype(int)

    for j in np.arange(0.0, NoEy, 1):
        for i in np.arange(0.0, NoEx, 1):
            n0 = i + j*NoNx
            conn.append([n0, n0 + 1, n0 + 1 + NoNx, n0 + NoNx])
    conn = np.array(conn).astype(int)
    plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    for i, txt in enumerate(nodes):
        plt.annotate(txt, (nodes[i, 0], nodes[i, 1]), fontsize=4, color='red')
    plt.title('Mesh')
    plt.axis('equal')
    fname = os.path.join(output_dir, f'{pltn}.png')
    plt.savefig(fname)
    plt.clf()
    return nodes, conn


def gradshape(xi):
    x, y = xi
    dN = np.array([[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],
                   [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]])
    return 0.25*dN


def Stiffness(TNoN, nodes, conn, gradshape, C):
    K = np.zeros((2*TNoN, 2*TNoN))
    q4 = np.array([[x/np.sqrt(3.0), y/np.sqrt(3.0)] for y in [-1.0, 1.0]
                   for x in [-1.0, 1.0]])
    B = np.zeros((3, 8))
    for c in conn:
        xIe = nodes[c, :]
        Ke = np.zeros((8, 8))
        for q in q4:
            dN = gradshape(q)
            J = np.dot(dN, xIe).T
            dN = np.dot(np.linalg.inv(J), dN)
            # Strain displacement B matrix  [3x8]
            B[0, 0::2] = dN[0, :]
            B[1, 1::2] = dN[1, :]
            B[2, 0::2] = dN[1, :]
            B[2, 1::2] = dN[0, :]
            Ke += np.dot(np.dot(B.T, C), B) * np.linalg.det(J)
        for i, I in enumerate(c):
            for j, J in enumerate(c):
                K[2*I, 2*J] += Ke[2*i, 2*j]
                K[2*I+1, 2*J] += Ke[2*i+1, 2*j]
                K[2*I+1, 2*J+1] += Ke[2*i+1, 2*j+1]
                K[2*I, 2*J+1] += Ke[2*i, 2*j+1]
    return K


def BC_Loads(TNoN, nodes, K, HeightMesh, WidthMesh, Force):
    f = np.zeros((2*TNoN))
    for i in range(TNoN):
        if nodes[i, 1] == 0.0:
            K[2*i, :] = 0.0
            K[2*i+1, :] = 0.0
            K[2*i, 2*i] = 1.0
            K[2*i+1, 2*i+1] = 1.0
        if nodes[i, 1] == HeightMesh:
            x = nodes[i, 0]
            f[2*i+1] = Force
            if x == 0.0 or x == WidthMesh:
                f[2*i+1] *= 0.5
    return K, f


def Solver(K, f):
    u = np.linalg.solve(K, f)
    return u


def Strain_Stress_Calc(u, NoNx, NoNy, EsizeX, EsizeY, nodes, conn, gradshape,
                       C):
    q4 = np.array([[x/np.sqrt(3.0), y/np.sqrt(3.0)] for y in [-1.0, 1.0]
                   for x in [-1.0, 1.0]])
    B = np.zeros((3, 8))
    ux = np.reshape(u[0::2], (NoNy, NoNx))
    uy = np.reshape(u[1::2], (NoNy, NoNx))
    xvec = np.zeros(NoNx*NoNy)
    yvec = np.zeros(NoNx*NoNy)
    res = np.zeros(NoNx*NoNy)
    node_strain = np.zeros((len(nodes), 3))
    node_stress = np.zeros((len(nodes), 3))
    for i in range(NoNx):
        for j in range(NoNy):
            xvec[i*NoNy+j] = i*EsizeX + ux[j, i]
            yvec[i*NoNy+j] = j*EsizeY + uy[j, i]
            res[i*NoNy+j] = uy[j, i]

    emin = np.full((3,), 9.0e9)
    emax = np.full((3,), -9.0e9)
    smin = np.full((3,), 9.0e9)
    smax = np.full((3,), -9.0e9)
    for c in conn:
        nodePts = nodes[c, :]
        for q in q4:
            dN = gradshape(q)
            J = np.dot(dN, nodePts).T
            dN = np.dot(np.linalg.inv(J), dN)
            B[0, 0::2] = dN[0, :]
            B[1, 1::2] = dN[1, :]
            B[2, 0::2] = dN[1, :]
            B[2, 1::2] = dN[0, :]
            UU = u[[2*c[0], 2*c[0] + 1, 2*c[1], 2*c[1] + 1, 2*c[2], 2*c[2] + 1,
                    2*c[3], 2*c[3] + 1]]
            strain = np.dot(B, UU)
            stress = np.dot(C, strain)
            emin = np.minimum(emin, strain)
            emax = np.maximum(emax, strain)
            node_strain[c, :] = strain.T
            node_stress[c, :] = stress.T
            smax = np.maximum(smax, stress)
            smin = np.minimum(smin, stress)
    return xvec, yvec, res, node_strain, node_stress


def postprocessing(nodes, u, node_stress, node_strain, plot_type, output_dir):
    xvec = nodes[:, 0] + u[::2]
    yvec = nodes[:, 1] + u[1::2]

    if plot_type == 'u1':
        res = u[::2]				# x-disp
    elif plot_type == 'u2':
        res = u[1::2]				# y-disp
    elif plot_type == 's11':
        res = node_stress[:, 0]  	# s11
    elif plot_type == 's22':
        res = node_stress[:, 1]  	# s22
    elif plot_type == 's12':
        res = node_stress[:, 2]  	# s12
    elif plot_type == 'e11':
        res = node_strain[:, 0]  	# e11
    elif plot_type == 'e22':
        res = node_strain[:, 1]  	# e22
    elif plot_type == 'e12':
        res = node_strain[:, 2]		# e12
    else:
        raise ValueError("Invalid plot type")

    t = plt.tricontourf(xvec, yvec, res, levels=14, cmap=plt.cm.jet)
    plt.grid()
    plt.colorbar(t)
    plt.title(plot_type)
    plt.axis('equal')
    fname = os.path.join(output_dir, f'{plot_type}.png')
    plt.savefig(fname, dpi=300)
    plt.clf()


ProcList = ['Plane_strain', 'Plane_stress']


def main():
    p = argparse.ArgumentParser(prog='FEM application to Plane Strain Plane'
                                'Stress problem')
    p.add_argument('-W', '--width', type=int, default=10)
    p.add_argument('-H', '--height', type=int, default=50)
    p.add_argument('-noex', type=int, default=10)
    p.add_argument('-noey', type=int, default=50)
    p.add_argument('--Mat', type=str)
    p.add_argument('-f', type=float)
    p.add_argument('--procedure', choices=ProcList,
                   help="Problem fromulation method")
    p.add_argument('--output-dir', type=str,
                   help='Output directory to generate file')
    args = p.parse_args()

    # Mesh Parameters
    NoEx = args.noex
    NoEy = args.noey
    WidthMesh = args.width
    HeightMesh = args.height
    Force = args.f

    # Derived Parameters
    NoNx = NoEx+1
    NoNy = NoEy+1
    TNoN = NoNx*NoNy
    EsizeX = WidthMesh/NoEx
    EsizeY = HeightMesh/NoEy
    # Material constitutive model - plane strain / plane stress
    E = float(args.Mat.split(',')[0])
    v = float(args.Mat.split(',')[1])

    if (args.procedure == 'Plane_strain'):
        C = E/(1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0],
                                              [v, 1.0-v, 0.0],
                                              [0.0, 0.0, 0.5-v]])
    else:
        C = E/(1-v**2)*np.array([[1, v, 0],
                                [v, 1, 0],
                                [0, 0, (1-v)/2]])
    # Mesh Creation
    Mesh = mesh_generate(HeightMesh, WidthMesh, NoNx, NoNy, NoEx, NoEy,
                         args.output_dir)
    nodes = Mesh[0]
    conn = Mesh[1]

    # Stiffness Matrix Creation
    sk = time.perf_counter()
    K = Stiffness(TNoN, nodes, conn, gradshape, C)
    time_takenk = time.perf_counter() - sk
    f = open('timeS.txt', 'w')
    f.write('timeS = ' + repr(time_takenk) + '\n')
    f.close()

    # Loads and Boundary condition applications
    LBC = BC_Loads(TNoN, nodes, K, HeightMesh, WidthMesh, Force)
    K = LBC[0]
    f = LBC[1]

    # Solution: Primary field variable Calculation
    u = Solver(K, f)
    print(E, v, max(u))
    fname = os.path.join(args.output_dir, 'disp.npz')
    np.savez(fname, u=u)

    # Derived field variable Calculation
    DFV = Strain_Stress_Calc(u, NoNx, NoNy, EsizeX, EsizeY, nodes, conn,
                             gradshape, C)
    node_strain = DFV[3]
    node_stress = DFV[4]
    fname = os.path.join(args.output_dir, 'strain.npz')
    np.savez(fname, node_strain=node_strain)
    fname = os.path.join(args.output_dir, 'stress.npz')
    np.savez(fname, node_stress=node_stress)

    # Postprocessing
    plot_type = ['u1', 'u2', 's11', 's22', 's12', 'e11', 'e22', 'e12']
    for plot in plot_type:
        postprocessing(nodes, u, node_stress, node_strain, plot,
                       args.output_dir)


if __name__ == '__main__':
    main()
