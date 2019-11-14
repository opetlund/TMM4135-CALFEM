# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np

def quad4e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a four node membrane element.

    :param list ex: element x coordinates [x1, x2, x3, x4]
    :param list ey: element y coordinates [y1, y2, y3, x4]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [8 x 8]
    :return mat fe: consistent load vector [8 x 1] (if eq!=None)
    """

    ex1 = ex[0:3]
    ex2 = ex[1:4]
    ey1 = ey[0:3]
    ey2 = ey[1:4]

    Ke = np.matrix(np.zeros((8,8)))
    fe = np.matrix(np.zeros((8,1)))

    # TODO: fill out missing parts (or reformulate completely)
    def quad4_shape_function(ksi, eta):
        #assume ksi = [ksi1, ksi2, ksi3, ksi4], and same for eta
        N4 = np.zeros(4)
        N4[0] = 0.25 * (1 + ksi)*(1 + eta)
        N4[1] = 0.25 * (1 - ksi)*(1 + eta)
        N4[2] = 0.25 * (1 + ksi)*(1 - eta)
        N4[3] = 0.25 * (1 - ksi)*(1 - eta)

        return N4
    
    N4matrix = np.zeros((2,8))
    # N4matrix =[[N4 , 0],
    #            [0 , N4]]
    N4matrix[0, :4] = N4
    N4matrix[1, 4:] = N4

    xyvector = np.zeros(8) 
    # [x1, x2, ... , y3, y4]
    xyvector[:4] = ex
    xyvector[4:] = ey

    xy = N4matrix * xyvector.T #vector xy = [[x(ksi, eta)], [y(ksi, eta)]]

    def offset(eu, ev):
        # assume u as [u1, u2, u3, u4], and same for v
        uvvector = np.zeros(8)
        uvvector[:4] = eu
        uvvector[4:] = ev
        # [u1, u2, ... , v3, v4]

        return N4matrix @ uvvector.T #vector uv = [[u(ksi, eta)], [v(ksi, eta)]]





    if eq is None:
        return Ke
    else:
        fe[:6,0]  += ft1
        fe[2:8,0] += ft2
        return Ke, fe



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





  