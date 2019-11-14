# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np


def plante(ex, ey, ep, D, eq=None):

    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')

    if ep[0] == 1:
        return tri3e(ex, ey, D, ep[1], eq)
    else:
        Dinv = np.inv(D)
        return tri3e(ex, ey, Dinv, ep[1], eq)


def tri3e(ex, ey, D, th, eq=None):
    """
    Compute the stiffness matrix for a two dimensional beam element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    tmp = np.matrix([[1, ex[0], ey[0]],
                     [1, ex[1], ey[1]],
                     [1, ex[2], ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area
    A = A2 / 2.0

    cyclic_ijk = [0, 1, 2, 0, 1]      # Cyclic permutation of the nodes i,j,k

    zeta_px, zeta_py = zeta_partials_x_and_y(ex, ey)

    B = np.mat(np.zeros((3, 6)))
    # Set columns of B
    for i in range(3):
        B[:, i * 2] = np.array([[zeta_px[i]], [0], [zeta_py[i]]])
        B[:, i * 2 + 1] = np.array([[0], [zeta_py[i]], [zeta_px[i]]])

    # Compute element stifness matrix
    Ke = A * th * B.T * D * B
    if eq is None:
        return Ke
    else:
        fx = A * th * eq[0] / 3.0
        fy = A * th * eq[1] / 3.0
        fe = np.mat([[fx], [fy], [fx], [fy], [fx], [fy]])
        return Ke, fe


def zeta_partials_x_and_y(ex, ey):
    """
    Compute partials of area coordinates with respect to x and y.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    """

    tmp = np.matrix([[1, ex[0], ey[0]],
                     [1, ex[1], ey[1]],
                     [1, ex[2], ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area

    cyclic_ijk = [0, 1, 2, 0, 1]      # Cyclic permutation of the nodes i,j,k

    zeta_px = np.zeros(3)           # Partial derivative with respect to x
    zeta_py = np.zeros(3)           # Partial derivative with respect to y

    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        zeta_px[i] = (ey[j] - ey[k]) / A2
        zeta_py[i] = (ex[k] - ex[j]) / A2

    return zeta_px, zeta_py

# Functions for 6 node triangle


def tri6_area(ex, ey):

    tmp = np.matrix([[1, ex[0], ey[0]],
                     [1, ex[1], ey[1]],
                     [1, ex[2], ey[2]]])

    A = np.linalg.det(tmp) / 2

    return A


def tri6_shape_functions(zeta):
    # Assumes zeta = [zeta1 zeta2 zeta3]
    cyclic_ijk = [0, 1, 2, 0, 1]      # Cyclic permutation of the nodes i,j,k

    N6 = np.zeros(6)

    for i in range(3):
        N6[i] = 2 * zeta[i] * (zeta[i] - 0.5)
        j = (i + 1) % 3
        N6[i + 3] = 4 * zeta[i] * zeta[j]

    return N6


def tri6_shape_function_partials_x_and_y(zeta, ex, ey):

    zeta_px, zeta_py = zeta_partials_x_and_y(ex, ey)

    N6_px = np.zeros(6)
    N6_py = np.zeros(6)

    cyclic_ijk = [0, 1, 2, 0, 1]      # Cyclic permutation of the nodes i,j,k

    for i in range(3):
        j = (i + 1) % 3
        N6_px[i] = (4 * zeta[i] - 1) * zeta_px[i]
        N6_py[i] = (4 * zeta[i] - 1) * zeta_py[i]
        N6_px[i + 3] = 4 * (zeta[i] * zeta_px[j] + zeta[j] * zeta_px[i])
        N6_py[i + 3] = 4 * (zeta[i] * zeta_py[j] + zeta[j] * zeta_py[i])
    return N6_px, N6_py


def tri6_Bmatrix_old(zeta, ex, ey):

    nx, ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.matrix(np.zeros((3, 12)))

    for i in range(6):
        Bmatrix[:, i * 2] = np.array([[nx[i]], [0], [ny[i]]])
        Bmatrix[:, i * 2 + 1] = np.array([[0], [ny[i]], [nx[i]]])

    return Bmatrix


def tri6_Jacobian_old(zeta, ex, ey):

    nx, ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)
    J = np.zeros((2, 2))
    for i in range(6):
        J[0, 0] += nx[i] * ex[i]
        J[0, 1] += nx[i] * ey[i]
        J[1, 0] += ny[i] * ex[i]
        J[1, 1] += ny[i] * ey[i]

    return 0.5 * np.abs(np.linalg.det(J))


def tri6_shape_functions_partials_ksi_and_ny(zeta):
    ksi = zeta[0]
    ny = zeta[1]
    N_ksi = np.zeros(6)
    N_ny = np.zeros(6)
    N_ksi[0] = 4 * ksi - 1
    N_ny[1] = 4 * ny - 1
    N_ksi[2] = -3 + 2 * ny + 4 * ksi
    N_ny[2] = -3 + 2 * ksi + 4 * ny
    N_ksi[3] = 4 * ny
    N_ny[3] = 4 * ksi
    N_ksi[4] = -4 * ny
    N_ny[4] = 4 - 4 * ksi - 8 * ny
    N_ksi[5] = 4 - 8 * ksi - 4 * ny
    N_ny[5] = -4 * ksi
    return N_ksi, N_ny


def tri6_Jacobian(P, X):
    J = np.matmul(P, X)
    return J


def tri6_Bmatrix(J, P):
    N_x_y = np.matmul(np.linalg.inv(J), P)
    B = np.zeros((3, 12))
    for i in range(6):
        # B[:, i * 2] = np.array([[N_x_y[0, i]], [0], [N_x_y[1, i]]])
        # B[:, i * 2 + 1] = np.array([[0], [N_x_y[1, i]], [N_x_y[0, i]]])
        B[:, i * 2] = np.array([N_x_y[0, i], 0, N_x_y[1, i]])
        B[:, i * 2 + 1] = np.array([0, N_x_y[1, i], N_x_y[0, i]])

    return B


def tri6_Kmatrix_old(ex, ey, D, th, eq=None):

    # zeta i values for numerical integration
    zetaInt = np.array([[0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])
    # Weights for numerical integration
    wInt = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    A = tri6_area(ex, ey)

    # Jacobianx

    Ke = np.matrix(np.zeros((12, 12)))

    # Numerical integration with Gauss quadrature for triangle element
    for i in range(len(wInt)):
        B = tri6_Bmatrix(zetaInt[:, i], ex, ey)
        Ke = Ke + wInt[i] * B.T * D * B * tri6_Jacobian(zetaInt[:, i], ex, ey)

    if eq is None:
        return Ke
    else:
        fe = np.matrix(np.zeros((12, 1)))

        # TODO: fill out missing parts (or reformulate completely)

        return Ke, fe


def tri6_Kmatrix(ex, ey, D, th, eq=None):

    # zeta i values for numerical integration
    zetaInt = np.array([[0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])
    # Weights for numerical integration
    wInt = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    A = tri6_area(ex, ey)

    # Jacobian

    Ke = np.matrix(np.zeros((12, 12)))

    X = np.zeros((6, 2))
    X[:, 0] = ex
    X[:, 1] = ey
    P = np.zeros((2, 6))
    print("X")
    print(X)
    # Numerical integration with Gauss quadrature for triangle element
    for i in range(len(wInt)):
        N_ksi, N_ny = tri6_shape_functions_partials_ksi_and_ny(zetaInt[:, i])
        P[0, :] = N_ksi
        P[1, :] = N_ny
        J = tri6_Jacobian(P, X)
        print("P")
        print(P)
        print("J")
        print(J)
        B = tri6_Bmatrix(J, P)
        Ke = Ke + wInt[i] * B.T * D * B * np.linalg.det(J)

    if eq is None:
        return Ke
    else:
        fe = np.matrix(np.zeros((12, 1)))

        # TODO: fill out missing parts (or reformulate completely)

        return Ke, fe


def tri6e(ex, ey, D, th, eq=None):
    return tri6_Kmatrix(ex, ey, D, th, eq)
