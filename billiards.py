import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


class Billiards():

    def __init__(self, boundary, x, v, poincare = False, convexity = 1000):
        # set up inputs
        self.xi = x
        self.vi = v/np.linalg.norm(v)
        self.boundary = boundary
        self.poincare = poincare
        self.N = convexity

        # lists to hold history of x,v,t,poincare
        self.x = [x]
        self.v = [v]
        self.t = []
        self.p = []

        self.nbounce = 0

        return

    def bounce(self):

        self._get_new_position()
        self.x.append(self.xi)

        self._get_new_velocity()
        self.v.append(self.vi)

        #optional returning of poincare section
        if self.poincare:
            v_tangent = np.sqrt(1 - np.dot(self.vi, self._normal_line())**2)
            self.p.append([self.t[-1], v_tangent])

        self.nbounce += 1

        return

    def plot_path(self, N = 1000):
        self.path_fig = plt.figure(figsize = (10, 10))

        T = np.linspace(0, 1, N)
        border = np.zeros([N, 2])
        for i in range(N):
            border[i] = self.boundary(T[i])

        plt.plot(border[:,0], border[:,1], color = 'black')
        plt.axis('equal')

        plt.scatter(self.x[0][0], self.x[0][1], color = 'gray', alpha = .2)

        for i in range(self.nbounce):
            plt.plot([self.x[i][0], self.x[i+1][0]], [self.x[i][1], self.x[i+1][1]],
                      color = 'red', alpha = (i+1)/self.nbounce)
        plt.title("Billiard Path", fontsize = 14)

        print("Path figure saved as self.path_fig")
        return

    def plot_poincare(self, alpha = 0.5, s = 5, cmap = 'coolwarm'):

        assert self.poincare != False, "Must have poincare history enabled to plot"
        assert self.p != [], "No poincare history has been saved"

        DELT = .1
        self.poincare_fig = plt.figure(figsize = (10,10))

        P = np.array(self.p)
        cgrad = np.arange(len(P))

        plt.scatter(P[:,0], P[:,1], cmap = cmap, c = cgrad, alpha = alpha, s = s)
        plt.xlim(0-DELT,1+DELT)
        plt.ylim(0-DELT,1+DELT)
        plt.xlabel('Location on Curve')
        plt.ylabel("Tangential Velocity")
        plt.title("Billiards Poincare Section", fontsize = 14)

        print("Poincare section figure saved as self.poincare_fig")
        return

    # computes the next pos and velocity of the ball after a bounce
    def _get_new_position(self):

        # np warning suppression
        if self.vi[0] == 0:
            m = np.inf
        else:
            m = self.vi[1]/self.vi[0]

        a = self.xi[0]
        b = self.xi[1]

        # uses gridsearch to find location of possible zeros
        T = np.linspace(0,1,self.N)
        obj_val = np.zeros(self.N)
        for i in range(self.N):
            obj_val[i] = self._border_dist(T[i], m, a, b)

        sign_change = (obj_val[:-1] * obj_val[1:]) < 0
        boundaries = np.where(sign_change==True)[0]

        #number of possible solutions given the spacing of T
        k = len(boundaries)

        solns = np.zeros(k)
        euc_dist = np.zeros(k)
        positions = np.zeros([k, 2])

        obj = lambda t: self._border_dist(t, m, a, b)

        # using possible zeros finds every zero along the line and their distance to the starting position
        for i in range(k):
            bracket = [T[boundaries[i]], T[boundaries[i]+1]]
            # bretnq should always converge like bisection.. if not change to bisect w/o much loss
            solns[i] = brentq(obj, bracket[0], bracket[1], xtol = 1e-14)
            positions[i] = self.boundary(solns[i])
            euc_dist[i] = np.linalg.norm(positions[i]-self.xi)

        # masks out solutions along the correct movement direction (uses/requires constant velocity)
        correct_dir = np.all(np.sign(positions - self.xi) == np.sign(self.vi), axis = 1)
        euc_dist[~correct_dir] = np.inf
        euc_dist[np.isclose(euc_dist, 0)] = np.inf

        # t values with the closest minimum
        t = solns[np.argmin(euc_dist)]
        self.t.append(t)
        self.xi = self.boundary(t)

        return

    def _get_new_velocity(self):

        #finds normalized normal line to the boundary edge hit
        n = self._normal_line()

        #finds vector between normal line and x-axis (using dot product)
        phi = np.arccos(np.abs(n[0]))
        #corrects rotation matrix for QI and QIII

        # np warning suppression
        if self.xi[1] != 0:
            if self.xi[0]/self.xi[1] < 0:
                phi = np.pi - phi

        # reflects previous velocity about the normal line to the boundary
        self.vi = -np.matmul(self._reflect(phi), self.vi)

        return

    # calculates distance between border and movement line for spot t on border
    def _border_dist(self, t, m, a, b):
        x, y = self.boundary(t)

        # gets rid of infinite/ large slopes to solve some conditioning problems
        if m > 1:
            dist = (1/m)*(y-b) + a - x
        else:
            dist = (m*(x-a) + b) - y

        return dist

    #calculate the normalized normal vector to the point boundary(t) using central difference formula for dydx
    def _normal_line(self, eps = 1e-6):
        n = ((self.boundary(self.t[-1]+eps) - self.boundary(self.t[-1]-eps))/(2*eps))[::-1]
        #returns unit vector tangent line
        return n/np.linalg.norm(n)


    #returns matrix for reflection around line theta from x axis
    @staticmethod
    def _reflect(theta):
        return np.array([[np.cos(theta)**2 - np.sin(theta)**2, 2*np.cos(theta)*np.sin(theta)],
                 [2*np.cos(theta)*np.sin(theta), np.sin(theta)**2 - np.cos(theta)**2]])

    # some basic boundaries

    @staticmethod
    def ellipse_boundary(s, a = np.sqrt(2), b = 1):
        s *= 2*np.pi
        return np.array([a*np.cos(s), b*np.sin(s)])
