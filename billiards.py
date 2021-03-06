import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping

#calculate the normalized normal vector to the point boundary(t) using central difference formula for dydx
def normal_line(s, boundary, eps = 1e-8):
    n = ((boundary(s+eps) - boundary(s-eps))/(2*eps))[::-1]
    #returns unit vector tangent line
    return n/np.linalg.norm(n)

def collide(ts, x, v, boundary):
    return np.linalg.norm(xy(ts[0], x, v) - boundary(ts[1]))

#parametrization of line between p1 and p2-- input t in [0,1] to get [x,y]
def xy(t, x, v):
    #start and end of the line segment (holds so long as boundary is inside unit square)
    p1 = x
    p2 = x + 3*v

    x = p1[0] +  t*(p2[0]-p1[0])
    y = p1[1] +  t*(p2[1]-p1[1])
    return np.array([x,y])

#reflect around line theta from x axis
def R(theta):
    return np.array([[np.cos(theta)**2 - np.sin(theta)**2, 2*np.cos(theta)*np.sin(theta)],
             [2*np.cos(theta)*np.sin(theta), np.sin(theta)**2 - np.cos(theta)**2]])

#function for calculating one billiard bounce
'''
Takes in a current position, velocity, and problem-specific information and returns the next location and velocity
    the billiard would bounce to on the boundary
x- 1d numpy array giving [x, y]
v- 1d numpy array giving [v_x, v_y]
boundary- parametrized boundary for the billiard to bounce in. function(t) returning [x,y] position on boundary as np array
tol- tolerance for the minimization routine to take into account. Determined minimization tolerance and number of basin hops preformed each call
poincare- boolean value for whether or not to return poincare section each bounce.
            if true returns: x_new, v_new, [poincare_section]
'''
def bounce(x, v, boundary, tol = 1e-8, poincare = False):
    #normalize s.t. velocity is direction unit vector
    v = v/np.linalg.norm(v)
    #Solve system of parametrized equations to find where the ball hits the boundary
    #Analytically solveable for some boundaries, but numerical solution works for any boundarie
    #t,s = minimize(collide, x0 = (0.58, .5), bounds = ((.01,1), (0,1)), args = (x,v,boundary), tol = 1e-8, method = 'Powell').x
    minargs = {'bounds': ((.01,1), (0,1)), 'args' : (x,v,boundary), 'tol': tol}
    #basin hops to preform relative to tolerance needed
    N = int((-np.log10(tol)))
    t,s = basinhopping(collide, x0=(.5,.5), niter = N, minimizer_kwargs = minargs).x
    m = boundary(s)
    #finds normalized normal line to the boundary edge hit
    n = normal_line(s, boundary)
    #finds vector between normal line and x-axis (using dot product)
    phi = np.arccos(np.abs(n[0]))
    #corrects for QI and QIII
    if m[0]/m[1] < 0:
        phi = np.pi - phi
    #new path from m
    v_new = -np.matmul(R(phi), v)
    #print(m, v_new)
    #optional returning of poincare section
    if poincare:
        v_tangent = np.sqrt(1 - np.dot(v, n)**2)
        return m, v_new, np.array([s, v_tangent])

    return m, v_new



# two basic boundary options
def circle(s):
    s *= 2*np.pi
    return np.array([np.cos(s),np.sin(s)])

def ellipse(s):
    s *= 2*np.pi
    return np.array([np.cos(s), (1/np.sqrt(2))*np.sin(s)])
