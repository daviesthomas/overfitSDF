# "a rudimentary way to visualize trained network surfaces"
import sys
sys.path.append('../')
import neuralImplicit.geometry as gm
import argparse
import tensorflow as tf
import os
import numpy as np

#HACK: igl and tensorflow link against OpenMP on osx. This is a workaround to allow it...
os.environ['KMP_DUPLICATE_LIB_OK'] = "1"

import matplotlib.pyplot as plt

def loadModel(modelPath, neuralKey=''):
    # LOAD THE MODEL
    #load serialized model
    if neuralKey == '':
        jsonFile = open(modelPath+'.json', 'r')
    else:
        jsonFile = open(neuralKey, 'r')

    sdfModel = tf.keras.models.model_from_json(jsonFile.read())
    jsonFile.close()
    #load weights
    sdfModel.load_weights(modelPath)
    #sdfModel.summary()

    return sdfModel

def inferSDF(sdfModel, res):
    # create data sequences
    cubeMarcher = gm.CubeMarcher()
    inferGrid = cubeMarcher.createGrid(res)
    S = sdfModel.predict(inferGrid)
    return S, inferGrid

def createAx(idx):
    subplot = plt.subplot(idx, projection='3d')
    subplot.set_xlim((-1,1))
    subplot.set_ylim((-1,1))
    subplot.set_zlim((-1,1))
    subplot.view_init(elev=10, azim=100)
    subplot.axis('off')
    subplot.dist = 8
    return subplot

def plotSamples(ax, pts, S, vmin = -1, is2d = False):
    

    #just show points inside the shape!
    mask = S < 0.0
    mask = np.squeeze(mask)
    print(mask)
    fS = S[mask]
    fPts = pts[mask]
    print(fPts.shape)
    x,y,z = np.hsplit(fPts,3)

    ax.scatter(x,y,z,c=fS, marker='.',cmap='coolwarm', norm=None, vmin=vmin, vmax=1)

if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='Neural Implicit mesher.')
    parser.add_argument('weightPath', type=str, help="path to neural implicit to be meshed, or folder of neural implicits")
    parser.add_argument('--outputPath', type=str,default='', help='destination path of generated meshes')
    parser.add_argument('--neuralKey', type=str, default='', help='path to neural implicit architecture json (the neural key)')
    parser.add_argument('--res', type=int,default=32, help='resolution of grid used in marching cubes')
    args = parser.parse_args()

    # support both single neural implicit, and a folder
    if os.path.isdir(args.weightPath):
        trainedModels = list([f.split('.')[0] for f in os.listdir(args.weightPath) if '.h5' in f])
        trainedModels = [os.path.join(args.weightPath, m) for m in trainedModels]
    else:
        trainedModels = [args.weightPath]

    # default to same location as weight path
    if (args.outputPath == ''):
        outputPath = args.weightPath
    else:
        outputPath = args.outputPath

    for m in trainedModels:
        try:
            print("[INFO] Loading model: ", m)
            sdfModel = loadModel(m, args.neuralKey)
            print("[INFO] Inferring sdf...")
            S, pts = inferSDF(sdfModel,args.res)
            print("[INFO] Plotting iso contour!")
            ax = createAx(111)
            plotSamples(ax, pts, S)
            plt.show()
            print("[INFO] Done.")
        except Exception as e:
            print (e)