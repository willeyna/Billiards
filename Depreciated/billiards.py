import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#function for calculating one billiard bounce
def bounce(x, v, boundary, d_boundary):
    #normalize s.t. velocity is direction unit vector
    v = v/np.linalg.norm(v)
    #minimize for spot along ball's path where it is closest to the wall (hits)
    #analytically solveable for circle and (probably) ellipse, but doing this allows for any boundary so long
    #    as you can describe the tangent line at any (x,y)
    t_min = minimize(param_boundary, 1/2, args = (x, v, boundary), bounds = ((.01,1),),method = 'Powell').x[0]
    #point where the path hits boundary; 2v used as second line segment point-- relies on normalization of boundaries
    m = xy(t_min, x, x + 2*v)
    #finds normalized normal line to the boundary edge hit
    n = np.array([1, -1/d_boundary(*m)])
    n /= np.linalg.norm(n)
    #finds vector between normal line and x-axis (using dot product)
    phi = np.arccos(n[0])
    #corrects for QI and QIII
    if m[0]/m[1] < 0:
        phi = np.pi - phi
    #new path from m
    v_new = -np.matmul(R(phi), v)

    return m, v_new

#parametrization of line between p1 and p2-- input t in [0,1] to get [x,y]
def xy(t, p1, p2):
    x = p1[0] +  t*(p2[0]-p1[0])
    y = p1[1] +  t*(p2[1]-p1[1])
    return np.array([x,y])

#distance to boundary parametrized
def param_boundary(t, x, v, boundary):
    return boundary(*xy(t, x, x + 2*v))

#reflect around line theta from x axis
def R(theta):
    return np.array([[np.cos(theta)**2 - np.sin(theta)**2, 2*np.cos(theta)*np.sin(theta)],
             [2*np.cos(theta)*np.sin(theta), np.sin(theta)**2 - np.cos(theta)**2]])

#gives error in equation
#move all nonzero terms to one side
#keep boundary within radius 1
def circle(x,y):
    return np.abs(x**2 + y**2 - 1)

#return dy/dx
def d_circle(x,y):
    return (-1*x)/y

def ellipse(x,y):
    return np.abs((x**2)/2 + y**2 - 0.5)

#return dy/dx
def d_ellipse(x,y):
    return (x)/(-2*y)
