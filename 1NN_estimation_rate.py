import numpy as np
import time
from codpy.plot_utils import multi_plot
from codpy.file import files_indir
import os,sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

from data_gen import *
from NN_map import *

def nn_estimation_rate(sampling_function, mapfnt, Ns, Ds, data, numTrials=10, N_sampling=10000, fun=None, method_name="COT"):
    
    L2_NN_Error = np.zeros((len(Ds), len(Ns), numTrials))
    times = {"data":[]}

    for k, d in enumerate(Ds):
        for i, n in enumerate(Ns):
            for t in range(numTrials):
                
                ### generate "training" data 
                source = sampling_function(n, d)
                source_ = sampling_function(n, d)
                target = mapfnt(source_)
                
                ### generate "L^2(P) estimation data" (out-of-sample points)
                source_mc = sampling_function(N_sampling, d)
                ot_t = mapfnt(source_mc)

                # Compute the transport using the specified method
                tic = time.time()
                tnn = fun(source_mc, source, target)
                toc = time.time()
                times['data'].append([d,n,numTrials,toc-tic])

                # Compute the L2 error
                L2_NN_Error[k, i, t] = (np.linalg.norm((tnn - ot_t), axis=1) ** 2).mean()

                if (t + 1) % numTrials == 0:
                    print('d=%f, n=%d, trials: %d/%d, time=%f' % (d, n, t + 1, numTrials, toc-tic))

        print('==== Done with dimension %f ====' % d)

    # Include the method name in the output file name
    pkl_title = 'CPU-{}_{}_error.pkl'.format(data, method_name)
    
    # Save the results to a file
    dict_data = {
        'Ds': Ds,
        'N_sampling': N_sampling,
        'Ns': Ns,
        'numTrials': numTrials,
        'L2_NN_Error': L2_NN_Error,
        'data': data,
        'times': times
    }
    
    with open(pkl_title, 'wb') as output:
        pickle.dump(dict_data, output)
    
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



def compare_methods(files, Ds, Ns, data, save=False):

    colors = ['r', 'b', 'g']
    linestyles = ['-', '--', '-.']

    def plot_MSE(id_d,ax=None,**kwargs):
        id,d = id_d[0], id_d[1]
        figsize = kwargs.get('figsize',(4, 4))
        if ax == None: fig, ax = plt.subplots(figsize=figsize)
        for method_id,file in enumerate(files):
            with open(file, 'rb') as f:
                file_data = pickle.load(f)

            l2_nn = file_data['L2_NN_Error']
            l2nn_d = l2_nn[id]

            file_name = os.path.basename(file)

            ynn_means = np.mean(l2nn_d, axis=-1)
            ynn_std = np.std(l2nn_d, axis=-1)

            x_ = sm.add_constant(np.log(np.array(Ns)))
            modelnn = sm.OLS(np.log(ynn_means), x_).fit()
            print(f'{file_name}, d={d}, rate={modelnn.params[1]}')

            plt.loglog(Ns, ynn_means, label=f'{file_name}', color=colors[method_id], linestyle=linestyles[method_id])
            plt.errorbar(Ns, ynn_means, yerr=ynn_std, color=colors[method_id], linestyle=linestyles[method_id])
            plt.title(kwargs.get("title",'d='+str(d)))
            plt.xlabel('$n$ samples')
            plt.ylabel('Mean Squared Error')

    def plot_time(id_d,ax=None,**kwargs):
        id,d = id_d[0], id_d[1]
        figsize = kwargs.get('figsize',(4, 4))
        if ax == None: fig, ax = plt.subplots(figsize=figsize)
        for method_id,file in enumerate(files):
            with open(file, 'rb') as f:
                file_data = pickle.load(f)

            l2_nn = pd.DataFrame(data = file_data['times']['data'], columns = ["D","N","K","times"])
            l2nn_d = (l2_nn[l2_nn["D"]==d]["times"]).to_numpy()
            l2nn_d = l2nn_d.reshape([len(Ns),int(l2nn_d.shape[0] / len(Ns))])

            file_name = os.path.basename(file)

            ynn_means = np.mean(l2nn_d, axis=-1)
            ynn_std = np.std(l2nn_d, axis=-1)

            x_ = sm.add_constant(np.log(np.array(Ns)))
            modelnn = sm.OLS(np.log(ynn_means), x_).fit()
            print(f'{file_name}, d={d}, rate={modelnn.params[1]}')

            plt.loglog(Ns, ynn_means, label=f'{file_name}', color=colors[method_id], linestyle=linestyles[method_id])
            plt.errorbar(Ns, ynn_means, yerr=ynn_std, color=colors[method_id], linestyle=linestyles[method_id])
            plt.title(kwargs.get("title",'d='+str(d)))
            plt.xlabel('$n$ samples')
            plt.ylabel('Execution Time')

    multi_plot(list(enumerate(Ds)),fun_plot=plot_MSE,mp_max_items=-1)
    plt.legend()
    plt.show()
    if save: plt.savefig(f'Error_{data}.pdf')

    multi_plot(list(enumerate(Ds)),fun_plot=plot_time,mp_max_items=-1)
    plt.legend()
    plt.show()
    if save: plt.savefig(f'Time_{data}.pdf')


if __name__ == "__main__":
    ### Example 1
    ### P = Unif(-1,1)^d
    ### T_0(x) = exp(x) coordinate-wise

    # Ds = [2,5,10]
    # Ns = [2**n for n in range(6,11)]
    Ds = [2,5,10,20,40]
    Ns = [2**n for n in range(6,11)]
    data = 'unif_exp'

    # DataViz(sample_uniform,OT_exp,500,2,N_sampling=500,fun = codpy_OT)

    # file_OTT = nn_estimation_rate(sample_uniform, OT_exp, Ns, Ds, data=data, fun=OTT, method_name="OTT")
    # file_POT = nn_estimation_rate(sample_uniform, OT_exp, Ns, Ds, data=data, fun=POT, numTrials=1,method_name="POT")
    # file_COT = nn_estimation_rate(sample_uniform, OT_exp, Ns, Ds, data=data, fun=COT,  method_name="COT")


    files = files_indir(os.path.dirname(os.path.realpath(__file__)),".pkl")



    compare_methods(
        files=files,
        Ds=Ds,
        Ns=Ns,
        data=data,
        save=True
    )
    pass