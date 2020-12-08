import numpy as np
import math
import tensorflow as tf

class SDFSequence(tf.keras.utils.Sequence):
  def __init__(self, sdf, pointSampler, batchSize, epochLength = 10**6, fourierMaxFreq=0):
    self._sdf = sdf
    self._epochLen = epochLength
    self._hasFrame = False  # configured below
    self.fourierMaxFreq = fourierMaxFreq

    #sdf can be either the sdf class or a precomputed array! 
    try:
      self._sdf.shape
      self._precomputed = True

      #[x,y,z,f,s]
      if self._sdf.shape[1] == 5:
        self._hasFrame = True
      
    except:
      self._precomputed = False
      assert(not pointSampler == None)
      self._pdf = pointSampler

    self._batchSize = batchSize

  def __len__(self):
    if self._precomputed:
      return math.ceil(self._sdf.shape[0]/self._batchSize)

    #arbitrary...
    return math.ceil(self._epochLen/self._batchSize)

  def __getitem__(self, idx):
    if self._precomputed:
      batch = self._sdf[idx * self._batchSize : (idx+1) * self._batchSize]

      if self._hasFrame:
        x,y,z,f,s = np.split(batch, 5, axis=1)
        X = np.squeeze(np.stack([x,y,z,f], axis=1))
      else:
        x,y,z,s = np.split(batch, 4, axis=1)
        X = np.squeeze(np.stack([x,y,z], axis=1))

      if (self.fourierMaxFreq > 0):
        bvals = 2.**np.arange(self.fourierMaxFreq/2)
        bvals = np.reshape(np.eye(3)*bvals[:,None,None], [len(bvals)*3, 3])
        avals = np.ones((bvals.shape[0])) 
        pts_flat = np.reshape(X, [-1,3])

        X = np.concatenate([avals * np.sin(pts_flat @ bvals.T), 
                    avals * np.cos(pts_flat @ bvals.T)], axis=-1)
        
      Y = np.array(s)
      return X, Y

    x = self._pdf.sample(self._batchSize)
    y = self._sdf.query(x)

    return np.array(x), np.array(y)