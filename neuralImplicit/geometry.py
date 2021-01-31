import sys
sys.path.insert(0, "../submodules/libigl/python/") 

from decimal import *
import math

import numpy as np 
import tensorflow as tf

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde

import matplotlib.cm as cm

import igl
import meshplot 

#constants used for sampling box AND miniball normalization
BOUNDING_SPHERE_RADIUS = 0.9
SAMPLE_SPHERE_RADIUS = 1.0

class SDF():
    def __init__(self, mesh):
        #TODO: support sign types.
        self.V = mesh.V
        self.F = mesh.F

    def query(self, queries):
        """Returns numpy array of SDF values for each point in queries"""
        S, _, _ = igl.signed_distance(queries, self.V, self.F, return_normals=False)
        S = np.expand_dims(S, axis=1)
        return S

class CubeMarcher():
    def __init__(self):
        pass 

    def march(self, grid, sdf):
        #TODO: marching cubes not yet implemented in libigl bindings, refactor once it is. Instead we use: https://anaconda.org/ilastik-forge/marching_cubes
        raise NotImplementedError("marching cubes not yet implemented...")
        #this doesn't work. Grid has to be reshaped to resxresxresx1 from res*res*resx3. SDF gives value at each grid cell! 
        v, _, f = marching_cubes.march(grid, 0)
        mesh = Mesh(V=v, F=f)
        return mesh

    def createGrid(self, res):
        K = np.linspace(
            -SAMPLE_SPHERE_RADIUS,
            SAMPLE_SPHERE_RADIUS,
            res
        )
        grid = [[x,y,z] for x in K for y in K for z in K]
        return np.array(grid)

class PointSampler(): 
    def __init__(self, mesh, ratio = 0.0, std=0.0, verticeSampling=False, importanceSampling=False):
        self._V = mesh.V
        self._F = mesh.F
        self._sampleVertices = verticeSampling

        if ratio < 0 or ratio > 1:
            raise(ValueError("Ratio must be [0,1]"))
        
        self._ratio = ratio
        
        if std < 0 or std > 1:
            raise(ValueError("Normal deviation must be [0,1]"))

        self._std = std

        self._calculateFaceBins()
    
    def _calculateFaceBins(self):
        """Calculates and saves face area bins for sampling against"""
        vc = np.cross(
            self._V[self._F[:, 0], :] - self._V[self._F[:, 2], :],
            self._V[self._F[:, 1], :] - self._V[self._F[:, 2], :])

        A = np.sqrt(np.sum(vc ** 2, 1))
        FA = A / np.sum(A)
        self._faceBins = np.concatenate(([0],np.cumsum(FA))) 

    def _surfaceSamples(self,n):
        """Returns n points uniformly sampled from surface of mesh"""
        R = np.random.rand(n)   #generate number between [0,1]
        sampleFaceIdxs = np.array(np.digitize(R,self._faceBins)) -1

        #barycentric coordinates for each face for each sample :)
        #random point within face for each sample
        r = np.random.rand(n, 2)
        A = self._V[self._F[sampleFaceIdxs, 0], :]
        B = self._V[self._F[sampleFaceIdxs, 1], :]
        C = self._V[self._F[sampleFaceIdxs, 2], :]
        P = (1 - np.sqrt(r[:,0:1])) * A \
                + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B \
                + np.sqrt(r[:,0:1]) * r[:,1:] * C

        return P

    def _verticeSamples(self, n):
        """Returns n random vertices of mesh"""
        verts = np.random.choice(len(self._V), n)
        return self._V[verts]
    
    def _normalDist(self, V):
        """Returns normal distribution about each point V"""
        if self._std > 0.0:
            return np.random.normal(loc = V,scale = self._std)

        return V
        
    def _randomSamples(self, n):
        """Returns n random points in unit sphere"""
        # we want to return points in unit sphere, could do using spherical coords
        #   but rejection method is easier and arguably faster :)
        points = np.array([])
        while points.shape[0] < n:
            remainingPoints = n - points.shape[0]
            p = (np.random.rand(remainingPoints,3) - 0.5)*2
            #p = p[np.linalg.norm(p, axis=1) <= SAMPLE_SPHERE_RADIUS]

            if points.size == 0:
                points = p 
            else:
                points = np.concatenate((points, p))
        return points

    def sample(self,n):
        """Returns n points according to point sampler settings"""

        nRandom = round(Decimal(n)*Decimal(self._ratio))
        nSurface = n - nRandom

        xRandom = self._randomSamples(nRandom)

        if nSurface > 0:
            if self._sampleVertices:
                # for comparison later :)
                xSurface = self._verticeSamples(nSurface)
            else:
                xSurface = self._surfaceSamples(nSurface)

            xSurface = self._normalDist(xSurface)
            if nRandom > 0:
                x = np.concatenate((xSurface,xRandom))
            else:
                x = xSurface
        else:
            x = xRandom

        np.random.shuffle(x)    #remove bias on order

        return x

class ImportanceSampler():
    # M, initital uniform set size, N subset size.
    def __init__(self, mesh, M, W):
        self.M = M # uniform sample set size
        self.W = W # sample weight...
    
        if (not mesh is None):
            #if mesh given, we can create our own uniform sampler
            self.uniformSampler = PointSampler(mesh, ratio=1.0) # uniform sampling
            self.sdf = SDF(mesh)
        else:
            # otherwise we assume uniform samples (and the sdf val) will be passed in.
            self.uniformSampler = None 
            self.sdf = None

    def _subsample(self, s, N):

        # weighted by exp distance to surface
        w = np.exp(-self.W*np.abs(s))
        # probabilities to choose each
        pU = w / np.sum(w)
        # exclusive sum
        C = np.concatenate(([0],np.cumsum(pU)))
        C = C[0:-1]

        # choose N random buckets
        R = np.random.rand(N)

        # histc
        I = np.array(np.digitize(R,C)) - 1

        return I


    ''' importance sample a given mesh, M uniform samples, N subset based on importance'''
    def sample(self, N):
        if (self.uniformSampler is None):
            raise("No mesh supplied, cannot run importance sampling...")

        #uniform samples
        U = self.uniformSampler.sample(self.M)
        s = self.sdf.query(U)
        I = self._subsample(s, N)

        R = np.random.choice(len(U), int(N*0.1))
        S = U[I,:]#np.concatenate((U[I,:],U[R, :]), axis=0)
        return S

    ''' sampling against a supplied U set, where s is sdf at each U'''
    def sampleU(self, N, U, s):
        I = self._subsample(s, N)
        q = U[I,:]
        d = s[I]
        return U[I,:], s[I] 

class Mesh():
    V = np.array([])
    F = np.array([])
    isNormalized = False

    def __init__(
        self, 
        meshPath=None, 
        V = None,
        F = None,
        doNormalize = True):

        if V is None and F is None:
            if meshPath is not None:
                self.V, self.F = igl.read_triangle_mesh(meshPath)
            else:
                raise UserWarning("Incorrect usage of Mesh class. Either path to existing mesh (meshPath) must be given, or array of vertices(V) and faces(F).")

        if doNormalize:
            self._normalizeMesh()

    def _normalizeMesh(self):
        # simply normalize vertices to fit in unit sphere
        centroid = np.mean(self.V, axis=0)
        self.V -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(self.V)**2, axis=1)))
        self.V /= furthest_distance

        self.isNormalized = True
        
    def show(self):
        meshplot.plot(self.V, self.F)

    def save(self, fp):
        igl.writeOBJ(fp,self._V,self._F)

if __name__ == '__main__':
    def main():
        meshplot.offline()
        
        import argparse
        import os
        parser = argparse.ArgumentParser(description='Geometry tools for overfitSDF')
        parser.add_argument('input_mesh', help='path to input mesh')

        args = parser.parse_args()

        mesh = Mesh(meshPath = args.input_mesh)

        # first test mesh is loaded correctly
        mesh.show()

        sdf = SDF(mesh)

        cubeMarcher = CubeMarcher()
        grid = cubeMarcher.createGrid(64)

        S = sdf.query(grid)


        #marchedMesh = cubeMarcher.march(grid, S)

        '''
        # test sdf sampling mesh
        sdf = SDF(mesh)

        cubeMarcher = CubeMarcher()
        grid = cubeMarcher.createGrid(64)

        S = sdf.query(grid)

        cubeMarcher.march(grid, S)
        marchedMesh = cubeMarcher.getMesh()

        marchedMesh.show()

        '''

    
    main()








