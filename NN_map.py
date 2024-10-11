import ot
import ott
import numpy as np
from sklearn.neighbors import NearestNeighbors
import codpy.core as core
from codpy.kernel import Kernel
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


#### 1-Nearest Neighbor Estimator
def LoopThruG(G,n,thr=1e-8):
  l1 = []
  l2 = []
  mx = G.max()
  for i in range(n):
    l1.append(i)
    for j in range(n):
      if G[i, j] / mx > thr:
        l2.append(j)
  return dict(zip(l1,l2))

def OT_0(source,target,a=None,b=None,maxiters=1000000):
    n = source.shape[0]
    if a == None:
        a = np.ones(n,)/n
    if b == None:
        b = np.ones(n,)/n
    M = ot.dist(source,target)
    G0 = ot.emd(a,b,M,numItermax=maxiters)
    return G0

def NNEstimator(x,source,target,G0,algo='brute'):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=algo).fit(source)
    _, indices = nbrs.kneighbors(x)

    #target_1nn = np.zeros((len(indices),dim))
    target_1nn = np.zeros_like(x)
    loopG = LoopThruG(G0,len(G0))
    for i, e in enumerate(indices):
      ind_ = loopG[int(e)]
      target_1nn[i] = target[ind_]

    return target_1nn

def POT(x,source,target):
    G0 = OT_0(source,target)
    return NNEstimator(x,source,target,G0)
   


def COT(x,source,target):
    # core.set_num_threads(1)
    return Kernel(set_kernel=core.kernel_setter("maternnorm","standardmean",0,1e-9)).map(source,target,distance="norm22")(x)

def COT_parallel(x,source,target):
    return Kernel(set_kernel=core.kernel_setter("maternnorm","standardmean",0,1e-9)).map(source,target,distance="norm22",sub=True)(x)

def OTT(source_mc, source, target, epsilon=None, max_iters=1000):
    
    # the point cloud geometry
    if epsilon is None:
       geom = pointcloud.PointCloud(source, target)
    else:
      geom = pointcloud.PointCloud(source, target, epsilon = epsilon)
    
    # solution of the OT problem
    problem = linear_problem.LinearProblem(geom)
    output = sinkhorn.Sinkhorn(max_iterations=max_iters)(problem)
    
    dual_potentials = output.to_dual_potentials()

    # transport_map#(out-of-sample points)
    transported_points = dual_potentials.transport(source_mc)
    
    return transported_points
