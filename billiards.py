import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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

# calculates distance between border and movement line for spot t on border
def border_dist(t, m, a, b, border):
    x, y = border(t)

    # gets rid of infinite/ large slopes to solve some conditioning problems
    if m > 1:
        dist = (1/m)*(y-b) + a - x
    else:
        dist = (m*(x-a) + b) - y

    return dist

# given pos and vel info calculates the location of the next collision
def collide(x, v, boundary, N=1000):

    # np warning suppression
    if v[0] == 0:
        m = np.inf
    else:
        m = v[1]/v[0]

    a = x[0]
    b = x[1]

    # uses gridsearch to find location of possible zeros
    T = np.linspace(0,1,N)
    obj_val = np.zeros(N)
    for i in range(N):
        obj_val[i] = border_dist(T[i], m, a, b, boundary)

    sign_change = (obj_val[:-1] * obj_val[1:]) < 0
    boundaries = np.where(sign_change==True)[0]

    #number of possible solutions given the spacing of T
    k = len(boundaries)

    solns = np.zeros(k)
    euc_dist = np.zeros(k)
    positions = np.zeros([k, 2])

    obj = lambda t: border_dist(t, m, a, b, boundary)

    # using possible zeros finds every zero along the line and their distance to the starting position
    for i in range(k):
        bracket = [T[boundaries[i]], T[boundaries[i]+1]]
        # bretnq should always converge like bisection.. if not change to bisect w/o much loss
        solns[i] = brentq(obj, bracket[0], bracket[1], xtol = 1e-14)
        positions[i] = boundary(solns[i])
        euc_dist[i] = np.linalg.norm(positions[i]-x)

    # masks out solutions along the correct movement direction (uses/requires constant velocity)
    correct_dir = np.all(np.sign(positions - x) == np.sign(v), axis = 1)
    euc_dist[~correct_dir] = np.inf

    # t values with the closest minimum
    final_t = solns[np.argmin(euc_dist)]
    final_x = boundary(final_t)

    return final_t, final_x

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
def bounce(x, v, boundary, poincare = False):

    t, m = collide(x, v, boundary)

    #finds normalized normal line to the boundary edge hit
    n = normal_line(t, boundary)

    #finds vector between normal line and x-axis (using dot product)
    phi = np.arccos(np.abs(n[0]))
    #corrects rotation matrix for QI and QIII

    # np warning suppression
    if m[1] != 0:
        if m[0]/m[1] < 0:
            phi = np.pi - phi

    v_new = -np.matmul(R(phi), v)

    #optional returning of poincare section
    if poincare:
        #normalize s.t. velocity is direction unit vector
        v = v/np.linalg.norm(v)

        v_tangent = np.sqrt(1 - np.dot(v, n)**2)
        return m, v_new, np.array([t, v_tangent])

    return m, v_new



# two basic boundary options
def circle(s):
    s *= 2*np.pi
    return np.array([np.cos(s),np.sin(s)])

def ellipse(s):
    s *= 2*np.pi
    return np.array([np.cos(s), (1/np.sqrt(2))*np.sin(s)])
