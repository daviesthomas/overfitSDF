import os
import sys
sys.path.insert(0, '..')

from train import singleModelTrain
from neuralImplicit.model import SDFModel, Config
import numpy as np
from neuralImplicit.geometry import PointSampler, Mesh, SDF
import pandas as pd

config = Config()

files = [os.path.join('../data/', f) for f in os.listdir('../data/')]

importanceWeights = range(0, 100, 15)

def trainModel(meshFile, config):
    if config.name in os.listdir('results/logs/'):
        return

    # train model on single mesh given
    singleModelTrain(
        meshFn=meshFile,
        precomputedFn=None,
        config=config,
        showVis=False
    )

for meshFile in files:
    # importance ablation
    for w in importanceWeights:
        config.name = os.path.splitext(os.path.basename(meshFile))[0] + '_{}'.format((w))
        config.samplingMethod = {
            'weight': w,
            'ratio': 0.1,
            'type': 'Importance'
        }
        trainModel(meshFile, config)

    # deepSDF style
    config.samplingMethod = {
      'std': 0.01,
      'ratio': 0.1,
      'type': 'SurfaceUniform'
    }
    config.name = os.path.splitext(os.path.basename(meshFile))[0] + '_surface'
    trainModel(meshFile, config)


    # uniform sampling
    config.samplingMethod = {
        'type':'Uniform'
    }
    config.name = os.path.splitext(os.path.basename(meshFile))[0] + '_uniform'
    trainModel(meshFile, config)

    # vertice sampling
    config.samplingMethod = {
      'std': 0.01,
      'ratio': 0.1,
      'type': 'Vertice'
    }
    config.name = os.path.splitext(os.path.basename(meshFile))[0] + '_vertice'
    trainModel(meshFile, config)


# generate uniform grid to eval all against
res = 64

K = np.linspace(
    -1.0,
    1.0,
    res
)
grid = [[x,y,z] for x in K for y in K for z in K]
uniformSamples = np.array(grid)

resultsDF = pd.DataFrame(columns=['mesh','importance','SD Uniform','SSD Uniform', 'SD Surface', 'SSD Surface'])

for meshFile in files:
    mesh = Mesh(meshFile)
    sdf = SDF(mesh)

    surfaceSampler = PointSampler(mesh)
    surfaceSamples = surfaceSampler.sample(n=64**3)

    # generate ground truth SDF in grid
    trueSDFUniform = sdf.query(uniformSamples)
    trueSDFSurface = sdf.query(surfaceSamples)

    for w in [str(i) for i in importanceWeights] + ['surface', 'uniform', 'vertice']:
        config.name = os.path.splitext(os.path.basename(meshFile))[0] + '_{}'.format((w))

        model = SDFModel(config)
        model.load()
        predSDFUniform = model.predict(uniformSamples)
        predSDFSurface = model.predict(surfaceSamples)

        res = {
            'mesh':meshFile,
            'importance':w,
            'Mean Uniform': np.mean(np.absolute(trueSDFUniform - predSDFUniform)),
            'SD Uniform': np.sum(np.absolute(trueSDFUniform - predSDFUniform)),
            'SSD Uniform': np.sum((trueSDFUniform - predSDFUniform)**2),
            'Mean Surface': np.mean(np.absolute(trueSDFSurface - predSDFSurface)),
            'SD Surface': np.sum(np.absolute(trueSDFSurface - predSDFSurface)),
            'SSD Surface': np.sum((trueSDFSurface-predSDFSurface)**2)
        }

        resultsDF = resultsDF.append(res, ignore_index=True)

    # also have to eval each of the prior art methods


resultsDF.to_csv('results.csv')









