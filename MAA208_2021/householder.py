#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:04:39 2022

@author: liuyufei
"""

from math import sqrt
import numpy as np


def householder(A):
    """Performs a Householder Reflections based QR Decomposition of the                                               
    matrix A. The function returns Q, an orthogonal matrix and R, an                                                  
    upper triangular matrix such that A = QR."""
    n = len(A)

    # Set R equal to A, and create Q as a zero matrix of the same size
    R = A
    Q = np.zeros((n,n))

    # The Householder procedure
    for k in range(n-1):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
        # Create identity matrix of same size as A                                                                    
        I = np.eye(n)

        # Create the vectors x, e and the scalar alpha
        # Python does not have a sgn function, so we use cmp instead
        print(A[:,0])
        x = A[:,k]
        e = I[:,k]
        
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        print(f"alpha: {alpha}, x : {x}, e: {e}")

        # Using anonymous functions, we create u and v
        u = x + alpha*e
        

        v = u/np.linalg.norm(u)
 

        # Create the Q minor matrix
        d = [[2*v[i] * v[j] for i in range(n-k)] for j in range(n-k) ]
        Q_min = np.eye(n-k) - d
        
        print(f"u: {u}, v : {v}, Q_min: {Q_min}")
        

        # "Pad out" the Q minor matrix with elements from the identity
        Q_t = Q.copy()
        for i in range(n):
            for j in range(n):
                if i<k or j<k:
                    Q_t[i][j] = I[i][j]
                else:
                    Q_t[i][j]=Q_min[i-k][j-k]

        # If this is the first run through, right multiply by A,
        # else right multiply by Q
        if k == 0:
            Q = Q_t
            R = np.dot(Q_t,A)
        else:
            Q = np.dot(Q_t,Q)
            R = np.dot(Q_t,R)

    # Since Q is defined as the product of transposes of Q_t,
    # we need to take the transpose upon returning it
    return Q.T, R

A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
Q, R = householder(A)

print("A:", A)
print("Q:", Q)
print("R:", R)
print("break")

import scipy
import scipy.linalg
A = scipy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])  # From the Wikipedia Article on QR Decomposition
Q, R = scipy.linalg.qr(A)
print("A:", A)
print("Q:", Q)
print("R:", R)


def gram_sm(A):

    n, m = A.shape # get the shape of A

    Q = np.empty((n, n)) # initialize matrix Q
    u = np.empty((n, n)) # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R

def QR_eigvals(A, tol=1e-12, maxiter=1000):
    "Find the eigenvalues of A using QR decomposition."

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = gram_sm(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals












