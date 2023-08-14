#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 11:39:48 2022

@author: npbhatt
"""
import numpy as np
import math
from sympy import *

alpha, beta, gamma = symbols('alpha beta gamma')

A = Matrix(
[[cos(alpha)*cos(beta),cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma),cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],
 [sin(alpha)*cos(beta),sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma),sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)],
 [-sin(beta),cos(beta)*sin(gamma),cos(beta)*cos(gamma)]]
 )

X = Matrix([[-7.03639746], [-6.85398054], [0.07615376]])
b = Matrix([[4.34303188], [-6.3880024], [6.1555047]])

X = X/X.norm()
b = b/b.norm()

print("--------- A ---------\n")
print(A)
print("\n")
print("--------- X ---------\n")
print(X)
print("\n")
print("--------- b ---------\n")
print(b)
print("\n")

print(b-A*X)

# solve(b-A*X, (alpha, beta, gamma), simplify=False)

# (0,-55*180/math.pi,90*180/math.pi)

print(A.subs([(alpha,-2),(beta,-55*180/math.pi),(gamma,90*180/math.pi)]))
