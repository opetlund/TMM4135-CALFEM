# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi, eta):
    
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    # TODO: fill inn values of the  shape functions
    N4 = np.zeros(4)
    N4[0] = 0.25 * (1 + xsi) * (1 + eta)
    N4[1] = 0.25 * (1 - xsi) * (1 + eta)
    N4[2] = 0.25 * (1 + xsi) * (1 - eta)
    N4[3] = 0.25 * (1 - xsi) * (1 - eta)
    return N4

def quad4_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi

    Ndxsi = np.zeros(4)
    Ndxsi[0] = 0.25 * (1 + eta)
    Ndxsi[1] = 0.25 * (-1 - eta)
    Ndxsi[2] = 0.25 * (1 - eta)
    Ndxsi[3] = 0.25 * (-1 + eta)
    return Ndxsi


def quad4_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    """
    # ----- Derivatives of shape functions with respect to eta -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi
    Ndeta = np.zeros(4)
    Ndeta[0] = 0.25 * (1 + xsi)
    Ndeta[1] = 0.25 * (1 - xsi)
    Ndeta[2] = 0.25 * (-1 - xsi)
    Ndeta[3] = 0.25 * (-1 + xsi)
    return Ndeta

def make_B_matrix(Nmatrix):
    Bmatrix = np.matrix(np.zeros((3, 8)))

    for i in range(4):
        Bmatrix[:, i * 2] = np.array([[Nmatrix[0, i]], [0], [Nmatrix[1, i]]])
        Bmatrix[:, i * 2 + 1] = np.array([[0], [Nmatrix[1, i]], [Nmatrix[0, i]]])
    return Bmatrix


def quad4e(ex, ey, D, thickness, eq=None):
    """
    Calculates the stiffness matrix for a 8 node isoparametric element in plane stress

    Parameters:

        ex  = [x1 ... x4]           Element coordinates. Row matrix
        ey  = [y1 ... y4]
        D   =           Constitutive matrix
        thickness:      Element thickness
        eq = [bx; by]       bx:     body force in x direction
                            by:     body force in y direction

    Returns:

        Ke : element stiffness matrix (8 x 8)
        fe : equivalent nodal forces (4 x 1)

    """
    t = thickness

    if eq is 0:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta


            #TODO: Calculate Jacobian, inverse Jacobian and determinant of the Jacobian
            J = np.matmul(G,H) #TODO: Correct this
            print("printing J: ", J)
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            Nmatrix = np.zeros((2,4))
            Nmatrix[0, :] = Ndxsi[:]
            Nmatrix[1, :] = Ndeta[:]

            Ndxy = np.matmul(invJ, Nmatrix) #Transform N-matrix derived wrt. xsi and eta to x and y
            B = make_B_matrix(Ndxy)


            #TODO: Fill out correct values for displacement interpolation xsi and eta
            N2 = np.zeros((2,8))
            N4matrix = np.zeros((2,8))
            # N4matrix =[[N4 , 0],
            #            [0 , N4]]
            N4matrix[0, :4] = quad4_shapefuncs(xsi, eta)
            N4matrix[1, 4:] = quad4_shapefuncs(xsi, eta)
            '''
            Not correct yet as we do not have eu anv ev
            def displacement(eu, ev):
                # make N2
                # assume eu as [u1, u2, u3, u4], and same for ev
                uvvector = np.zeros(8)
                uvvector[:4] = eu
                uvvector[4:] = ev
                # [u1, u2, ... , v3, v4]

                return N4matrix @ (uvvector.T) #2x8-vector N2 = [[u(ksi, eta)], [v(ksi, eta)]]
            '''

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            #fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    return Ke, fe  # Returns stiffness matrix and nodal force vector


def quad9e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a four node membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    Ke = np.matrix(np.zeros((18,18)))
    fe = np.matrix(np.zeros((18,1)))

    # TODO: fill out missing parts (or reformulate completely)

    if eq is None:
        return Ke
    else:
        return Ke, fe





  