# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np

def plante(ex,ey,ep,D,eq=None):
    
    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')
        
    if ep[0] == 1 :
        return tri3e(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return tri3e(ex,ey,Dinv,ep[1],eq)


def tri3e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a two dimensional beam element.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: element stiffness matrix [6 x 1] (if eq!=None)
    """
    
    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(tmp)  # Double of triangle area
    A  = A2 / 2.0
       
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k
    
    zi_px = np.zeros(3)           # Partial derivative with respect to x
    zi_py = np.zeros(3)           # Partial derivative with respect to y
    
    for i in range(3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        zi_px[i] = (ey[j] - ey[k]) / A2
        zi_py[i] = (ex[k] - ex[j]) / A2
        
    B = np.matrix([
            [zi_px[0],        0, zi_px[1],        0, zi_px[2],       0],
            [       0, zi_py[0],        0, zi_py[1],        0,zi_py[2]],
            [zi_py[0], zi_px[0], zi_py[1], zi_px[1], zi_py[2], zi_px[2]]])
    
    Ke = (B.T * D * B) * A * th
      
    if eq is None:
        return Ke
    else:
        fx = A * th * eq[0]/ 3.0
        fy = A * th * eq[1]/ 3.0
        fe = np.mat([[fx],[fy],[fx],[fy],[fx],[fy]])
        return Ke, fe
    