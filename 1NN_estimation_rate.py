import numpy as np
import time

import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

from data_gen import *
from NN_map import *

def nn_estimation_rate(sampling_function,mapfnt,Ns,Ds,data,numTrials=10,N_sampling=10000,fun = COT):  
    L2_NN_Error = np.zeros( (len(Ds), len(Ns), numTrials))
    tstart = time.time()
    for k, d in enumerate(Ds):
        for i, n in enumerate(Ns):
            for t in range(numTrials):
                
                ### generate ``training" data 
                source = sampling_function(n,d)
                source_ = sampling_function(n,d)
                target = mapfnt(source_)
                
                ### generate "L^2(P) estimation data"
                source_mc  = sampling_function(N_sampling,d)
                ot_t = mapfnt(source_mc)

                tnn = fun(source_mc,source,target)

                L2_NN_Error[k,i,t] = (np.linalg.norm( (tnn - ot_t),axis=1)**2).mean()

                if (t+1)%numTrials == 0:
                    print('d=%f, n=%d, trials: %d/%d, time=%d'%(d, n,t+1, numTrials, time.time()-tstart))
        print('==== Done %f d level ===='%(d))

    dict_data = {'Ds': Ds,
                 'N_sampling': N_sampling,
                 'Ns': Ns,
                 'numTrials': numTrials,
                 'L2_NN_Error': L2_NN_Error,
                 'data': data}

    pkl_title = 'CPU-{}_error.pkl'.format(data)
    output = open(pkl_title,'wb')
    pickle.dump(dict_data,output)
    output.close()
    return pkl_title
    
def CreatePlot_CPU_Errors(file,save=False):
    with open(file, 'rb') as f:
        file1 = pickle.load(f)

    l2_nn = file1['L2_NN_Error']

    Ds = file1['Ds']
    Ns = file1['Ns']
    data = file1['data']
    
    plt.figure(figsize=([6,5]))
    cz = ['r', 'b', 'g']
    print('Estimation rates for 1NN estimator')
    print('Note: theoretical rate is (-2/d)')
    for i,d in enumerate(Ds):

        c = cz[i]
        l2nn_d = l2_nn[i]

        ynn_means = np.mean(l2nn_d,axis=-1)
        ynn_std = np.std(l2nn_d,axis=-1)

        x_ = sm.add_constant(np.log(np.array(Ns)))
        modelnn = sm.OLS(np.log(ynn_means), x_).fit()
        print('d={}, rate={}'.format(d,modelnn.params[1]))

        plt.loglog(Ns,ynn_means,label='d={}'.format(d),c=c)
        plt.errorbar(Ns,ynn_means,yerr=ynn_std,c=c)

    plt.legend()
    plt.xlabel('$n$ samples')
    plt.ylabel('Mean Squared Error')
    if save == True:
        plt.savefig('cpu_error_comp_{}.pdf'.format(data))
    else:
        plt.show()

def DataViz(sampling_function,mapfnt,N,D,N_sampling=10000,fun = COT):
    source = sampling_function(N,D)
    source_copy = sampling_function(N,D)
    target = mapfnt(source_copy)

    #new samples to test estimators
    x = sampling_function(N_sampling,D)
    y_x = mapfnt(x)

    nn_x = fun(x,source,target)

    fig = plt.figure(dpi=100)
    plt.scatter(x[:,0],x[:,1],c='r',marker='.',label='Source')
    plt.scatter(y_x[:,0],y_x[:,1],edgecolor='b',facecolor='none',label='Target')
    plt.scatter(nn_x[:,0],nn_x[:,1],c='m',marker='+',label='1NN map')
    # plt.plot(x.T, nn_x.T, color="black", linewidth=.5)
    # plt.plot(y_x[:,0],y_x[:,1], nn_x[:,0],nn_x[:,1], color="black", linewidth=.5)
    plt.legend()
    plt.show()        


if __name__ == "__main__":
    ### Example 1
    ### P = Unif(-1,1)^d
    ### T_0(x) = exp(x) coordinate-wise

    Ds = [2,5,10]
    Ns = [2**n for n in range(6,11)]
    data = 'unif_exp'

    DataViz(sample_uniform,OT_exp,500,2,N_sampling=500,fun = codpy_OT)
    # DataViz(sample_uniform,OT_exp,500,2,N_sampling=20,fun = COT)

    file = nn_estimation_rate(sample_uniform,OT_exp,Ns,Ds,data=data,fun = COT)    
    CreatePlot_CPU_Errors(file)
    file = nn_estimation_rate(sample_uniform,OT_exp,Ns,Ds,data=data,fun = codpy_OT)    
    CreatePlot_CPU_Errors(file)
    pass