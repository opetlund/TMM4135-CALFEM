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

    # Get partial derivatives of zeta / area coordinates
    zeta_px, zeta_py = zeta_partials_x_and_y(ex, ey)

    # Initiate B to zeros
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
        # Distributed load on element
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

    zeta_px = np.zeros(3)           # Partial derivative with respect to x
    zeta_py = np.zeros(3)           # Partial derivative with respect to y

    for i in range(3):
        # Cyclic permutation
        j = (i + 1) % 3
        k = (i + 2) % 3

        # Set derivatives
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

    # Interpolation polynomial vector
    N6 = np.zeros(6)

    for i in range(3):
        # Set corner interpolation polynomial
        N6[i] = 2 * zeta[i] * (zeta[i] - 0.5)

        # Cyclic permutation
        j = (i + 1) % 3

        # Edge interpolation polynomial
        N6[i + 3] = 4 * zeta[i] * zeta[j]

    return N6


def tri6_shape_function_partials_x_and_y(zeta, ex, ey):

    # Get zeta partial derivatives
    zeta_px, zeta_py = zeta_partials_x_and_y(ex, ey)

    # Initiate vectors for interpolation polynomial derivatives vectors
    N6_px = np.zeros(6)
    N6_py = np.zeros(6)

    for i in range(3):
        # Cyclic permutation
        j = (i + 1) % 3

        # Set corner interpolation polynomial derivatives
        N6_px[i] = (4 * zeta[i] - 1) * zeta_px[i]
        N6_py[i] = (4 * zeta[i] - 1) * zeta_py[i]

        # Set edge interpolation polynomial derivatives
        N6_px[i + 3] = 4 * (zeta[i] * zeta_px[j] + zeta[j] * zeta_px[i])
        N6_py[i + 3] = 4 * (zeta[i] * zeta_py[j] + zeta[j] * zeta_py[i])

    return N6_px, N6_py


def tri6_Bmatrix(zeta, ex, ey):

    # Get interpolation polynomial derivatives
    nx, ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    # Initiate B matrix
    Bmatrix = np.matrix(np.zeros((3, 12)))

    for i in range(6):
        # Set columns of B
        Bmatrix[:, i * 2] = np.array([[nx[i]], [0], [ny[i]]])
        Bmatrix[:, i * 2 + 1] = np.array([[0], [ny[i]], [nx[i]]])

    return Bmatrix


def tri6_Kmatrix(ex, ey, D, th, eq=None):

    # zeta i values for numerical integration
    zetaInt = np.array([[0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])

    # Weights for numerical integration
    wInt = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    # Area of element
    A = tri6_area(ex, ey)

    # Initiate results
    Ke = np.matrix(np.zeros((12, 12)))
    fe = np.matrix(np.zeros((12, 1)))

    # Numerical integration with Gauss quadrature for triangle element
    for i in range(len(wInt)):

        # current zeta values and weight
        zeta = zetaInt[:, i]
        weight = wInt[i]

        # Get B matrix
        B = tri6_Bmatrix(zeta, ex, ey)

        # Add to Ke
        Ke = Ke + weight * A * th * (B.T @ D @ B)

        if eq is not None:
            # Distributed forces vector
            fvec = np.array([[eq[0]], [eq[1]]])

            # interpolation polynomial vector
            N6 = tri6_shape_functions(zeta)

            # Initiate N* matrix
            N2mat = np.zeros((2, 12))

            for j in range(6):
                # Place interpolation polynomials in N*
                N2mat[0, j * 2] = N6[j]
                N2mat[1, j * 2 + 1] = N6[j]

            # Add to fe
            fe += N2mat.T @ fvec * A * weight * th

    if eq is None:
        return Ke
    else:
        return Ke, fe


def tri6e(ex, ey, D, th, eq=None):
    return tri6_Kmatrix(ex, ey, D, th, eq)
