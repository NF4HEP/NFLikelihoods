import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
from scipy import stats
import sys
sys.path.append('../code')
import Distributions,Bijectors
#import Trainer_2 as Trainer
import Metrics as Metrics
from statistics import mean,median
import pickle
import  matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer
import os
import math


import GenerativeModelsMetrics as GMetrics
#import CorrelatedGaussians
#import Plotters
#import MixtureDistributions
#import compare_logprobs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import PlotsandHDPIforReload as PlotsandHDPI
import matplotlib.lines as mlines
#import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils

def loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file):


        #### import target distribution ######
        
        
        ###pickle train
        print("Importing X_data_train from file",X_data_train_file)
        pickle_train = open(X_data_train_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_train_file)
        X_data_train = pickle.load(pickle_train)
        print(np.shape(X_data_train))
        pickle_train.close()
  
  
        ###pickle test
        print("Importing X_data_test from file",X_data_test_file)
        pickle_test = open(X_data_test_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_test_file)
        X_data_test = pickle.load(pickle_test)
        print(np.shape(X_data_test))
        pickle_test.close()
        
        
        ###pickle test
        print("Importing logprobs_data_test from file",logprobs_data_test_file)
        pickle_logprobs = open(logprobs_data_test_file,'rb')
        start = timer()
        statinfo = os.stat(X_data_test_file)
        logprobs_data_test = pickle.load(pickle_logprobs)
        print(np.shape(logprobs_data_test))
        pickle_logprobs.close()

        end = timer()
        
        print('Files loaded in ',end-start,' seconds.\nFile size is ',statinfo.st_size,'.')
        return X_data_train,X_data_test,logprobs_data_test


def load_hyperparams(path_to_results):

    hyperparams_path=path_to_results+'/hyperparams.txt'
    hyperparams_frame=pd.read_csv(hyperparams_path)
    lastone=int(hyperparams_frame.shape[0]-1)
    print('lastone')
    print(lastone)
    ndims=int(hyperparams_frame['ndims'][lastone])
    nsamples=int(hyperparams_frame['nsamples'][lastone])
    bijector_name=str(hyperparams_frame['bijector'][lastone])
    nbijectors=int(hyperparams_frame['nbijectors'][lastone])
    batch_size=int(hyperparams_frame['batch_size'][lastone])
    spline_knots=int(hyperparams_frame['spline_knots'][lastone])
    range_min=int(hyperparams_frame['range_min'][lastone])
    activation=str(hyperparams_frame['activation'][lastone])
    regulariser=str(hyperparams_frame['regulariser'][lastone])
    eps_regulariser=float(hyperparams_frame['eps_regulariser'][lastone])
    hllabel=str(hyperparams_frame['hidden_layers'][lastone])
    
    
    hidden_layers=hllabel.split('-')
    for i in range(len(hidden_layers)):
        hidden_layers[i]=int(hidden_layers[i])

    return ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser




def ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser):


    if regulariser=='l1':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
    if regulariser=='l2':
        regulariser=tf.keras.regularizers.l2(eps_regulariser)
    else:
        regulariser=None
    
        

    #if bijector_name=='CsplineN':
    #    rem_dims=int(ndims/2)
    #    bijector=Bijectors.CsplineN(ndims,rem_dims,spline_knots,nbijectors,range_min,hidden_layers,activation)
    
    if bijector_name=='MsplineN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser)
        


    if bijector_name=='MAFN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser,perm_style='reverse')
        
    if bijector_name=='RealNVPN':
        rem_dims=int(ndims/2)
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.RealNVPN(ndims,rem_dims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)

    return bijector
    

def create_flow(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser):
    
    bijector=ChooseBijector(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser)
    base_dist=Distributions.gaussians(ndims)
    nf_dist=tfd.TransformedDistribution(base_dist,bijector)

    return nf_dist

def load_model(nf_dist,path_to_results,ndims,lr=.00001):


    x_ = Input(shape=(ndims,), dtype=tf.float32)
    print(x_)
    print(nf_dist)
    log_prob_ = nf_dist.log_prob(x_)
    print('hello')
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'/model_checkpoint/weights')

    return nf_dist,model
    

        
@tf.function
def save_iter(nf_dist,sample_size,iter_size,n_iters):
    #first iter
    sample_all=nf_dist.sample(iter_size)
    for j in range(1,n_iters):
        if j%100==0:
            print(j/n_iters)
            #print(tf.shape(sample_all))
        sample=nf_dist.sample(iter_size)

        #sample=postprocess_data(sample,preprocess_params)
        sample_all=tf.concat([sample_all,sample],0)
        #if j%1==0:
        #    with open(path_to_results+'/nf_sample_5_'+str(j)+'.npy', 'wb') as f:
        #        np.save(f, sample, allow_pickle=True)
        #tf.keras.backend.clear_session()
    return sample_all



def save_sample(nf_dist,path_to_results,sample_size=100000,iter_size=10000):
    print('saving samples...')
    n_iters=int(sample_size/iter_size)
    
 

    sample_all=save_iter(nf_dist,sample_size,iter_size,n_iters)
    sample_all=sample_all.numpy()
    print('hello')

    #print(np.shape(sample_all))
    with open(path_to_results+'/nf_sample_200k_sf.npy', 'wb') as f:
        np.save(f, sample_all, allow_pickle=True)
    print('samples saved')
    return
        
def load_sample(path_to_results):

    nf_sample=np.load(path_to_results+'/nf_sample.npy',allow_pickle=True)
    #nf_sample=np.load(path_to_results+'/sample_nf.pcl',allow_pickle=True)
    
    
    return nf_sample
        



def retrain_model(model,X_data,n_epochs,batch_size,patience=50,min_delta_patience=.00001,n_disp=1):

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns


    #earlystopping
    early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=min_delta_patience, patience=patience, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
     )
    # Display the loss every n_disp epoch
    epoch_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
                        print('\n Epoch {}/{}'.format(epoch+1, n_epochs, logs),
                              '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                       if epoch % n_disp == 0 else False
    )


    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    path_to_results+'/model_checkpoint/weights',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
    options=None,

                )
                
                
    StopOnNAN=tf.keras.callbacks.TerminateOnNaN()



    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.3,
                        shuffle=True,
                        verbose=2,
                        callbacks=[epoch_callback,early_stopping,checkpoint,StopOnNAN])
    return history,nf_dist

'''

def ComputeMetrics(X_data_test,nf_dist):

    #kl_divergence=Metrics.KL_divergence(X_data_test,nf_dist,test_log_probs)
    kl_divergence=-1
    ks_test_list=Metrics.KS_test(X_data_test,nf_dist)
    ks_median=median(ks_test_list)
    ks_mean=mean(ks_test_list)
    
    w_distance_list=Metrics.Wasserstein_distance(X_data_test,nf_dist)
    w_distance_median=median(w_distance_list)
    w_distance_mean=median(w_distance_list)
    
    frob_norm,nf_corr,target_corr=Metrics.FrobNorm(X_data_test,nf_dist)
    
    return kl_divergence,ks_median,ks_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr

def ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,w_distance_median,w_distance_mean,frob_norm,hidden_layers,batch_size,eps_regulariser,regulariser):

            results_dict.get('ndims').append(ndims)
            results_dict.get('bijector').append(bijector_name)
            results_dict.get('nbijectors').append(nbijectors)
            results_dict.get('nsamples').append(nsamples)
            results_dict.get('activation').append(activation)
            results_dict.get('spline_knots').append(spline_knots)
            results_dict.get('range_min').append(range_min)
            results_dict.get('kl_divergence').append(kl_divergence)
            results_dict.get('ks_test_mean').append(ks_mean)
            results_dict.get('ks_test_median').append(ks_median)
            results_dict.get('Wasserstein_median').append(w_distance_median)
            results_dict.get('Wasserstein_mean').append(w_distance_mean)
            results_dict.get('frob_norm').append(frob_norm)
            results_dict.get('time').append(-1)
            results_dict.get('hidden_layers').append(hidden_layers)
            results_dict.get('batch_size').append(batch_size)
            results_dict.get('eps_regulariser').append(eps_regulariser)
            results_dict.get('regulariser').append(regulariser)

            return results_dict
'''
def results_current(path_to_results,results_dict):

    currrent_results_file=open(path_to_results+'results_reload_test.txt','w')
    header=','.join(list(results_dict.keys()))



    currrent_results_file.write(header)
    currrent_results_file.write('\n')
    
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
    
    string=','.join(string_list)
    currrent_results_file.write(string)
    currrent_results_file.write('\n')
    
    currrent_results_file.close()


    return


def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    return preprocess_params


def postprocess_data(data,preprocess_params):

    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    postprocess_data=data*stds+means
    
    return postprocess_data


def preprocess_data(data,preprocess_params):
    
    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    preprocess_data=(data-means)/stds
    
    return preprocess_data


def MeansAndStdMetrics(metric_result):

    metric_mean=np.mean(metric_result)
    print(metric_mean)
    metric_std=np.std(metric_result)

    return metric_mean,metric_std
    
def SaveNewMetrics(metrics_dict,metric_name,results_path):

    name=results_path+'/'+metric_name+'_dict_sf'
    np.save(name,metrics_dict)


    return


def SaveResultsCurrentNewMetrics(results_dict_newmetrics,path_to_results):

    
    Frame=pd.DataFrame(results_dict_newmetrics)
    current_row= Frame.tail(1)
    current_row.to_csv(path_to_results+'/results_new_metrics_sf.txt',index=False)

    return


def SaveMeans(test_means):

    print(pd.DataFrame(test_means))

    return


def GetReusltsForPOI(results_list,labels,path_to_results,name):

    results_list=np.transpose(results_list)
    print(results_list)
    
    POI_results_mean=np.mean(results_list,axis=1)
    print(POI_results_mean)
    POI_results_mean_frame=pd.DataFrame(POI_results_mean).transpose()
    POI_results_mean_frame.columns =labels
    POI_results_mean_frame.to_csv(path_to_results+'/'+name+'_POI_results_sf.txt',index=False)
    print(POI_results_mean_frame)
    return POI_results_mean.tolist()

def ResultsToDict_NewMetrics(KS_pv_mean,KS_pv_std,KS_st_mean,KS_st_std,SWDmean,SWDstd):
    """
    Function that writes results to the a dictionary.
    """
    results_dict_newmetrics.get('KS_pv_mean').append(KS_pv_mean)
    results_dict_newmetrics.get('KS_pv_std').append(KS_pv_std)
    results_dict_newmetrics.get('KS_st_mean').append(KS_st_mean)
    results_dict_newmetrics.get('KS_st_std').append(KS_st_std)

    results_dict_newmetrics.get('SWD_mean').append(SWDmean)
    results_dict_newmetrics.get('SWD_std').append(SWDstd)

    return results_dict_newmetrics
    
def SoftClipTransform(nf_dist,hinge):

    max_conditions=[4.327962, 3.7579556, -0.9798687, 2.2252467, 4.658059, 4.31588, 4.4592004, 3.9522023, 4.261921, 4.3781595, 6.3621454, 6.0448027, 4.1009293, 4.318612, 4.198794, 4.241501, 4.8653646, 4.35286, 4.4388275, 4.3903165, 4.1843567, 4.4151235, 4.498217, 1.1032621, 3.6203547, 3.0772672, 4.4469132, 4.685502, -0.16419072, 1.3842824, 3.1064844, 0.18638913, 4.456744, 4.498217, 2.3278253, 4.6833744, 4.111755, 4.4213786, 4.444342, 4.2257795]


    min_conditions=[-3.9522753, -2.2016175, -1.0193546, -0.28156346, -3.292229, -3.8847764, -4.358234, -4.178113, -3.8904715, -3.940681, -1.3261361, -1.332359, -4.5354037, -4.8161125, -4.4008904, -4.188911, -4.430316, -4.549724, -4.2016883, -5.249503, -4.360835, -4.4719305, -4.694073, 0.88552004, -2.3991446, -4.376896, -4.21204, -4.754149, -1.7580823, -2.8867307, -4.4794407, -2.162731, -4.7337117, -4.694073, -3.7768517, -4.7883673, -4.2380314, -4.580329, -4.2114873, -4.1991444]
 
    bijector=tfb.SoftClip(low=min_conditions, high=max_conditions,hinge_softness=hinge)

    nf_dist=tfd.TransformedDistribution(nf_dist,bijector)

    return nf_dist
def CornerPlotter(target_samples,nf_samples,path_to_plot,selection_list):


    labels=[r'$\mu$',r'$\delta_{1}$',r'$\delta_{2}$',r'$\delta_{3}$',r'$\delta_{4}$',r'$\delta_{5}$',r'$\delta_{6}$',r'$\delta_{7}$',r'$\delta_{8}$',r'$\delta_{9}$',r'$\delta_{10}$',r'$\delta_{11}$',r'$\delta_{12}$',r'$\delta_{13}$',r'$\delta_{14}$',r'$\delta_{15}$',r'$\delta_{16}$',r'$\delta_{17}$',r'$\delta_{18}$',r'$\delta_{19}$',r'$\delta_{20}$',r'$\delta_{21}$',r'$\delta_{22}$',r'$\delta_{23}$',r'$\delta_{24}$',r'$\delta_{25}$',r'$\delta_{26}$',r'$\delta_{27}$',r'$\delta_{28}$',r'$\delta_{29}$',r'$\delta_{30}$',r'$\delta_{31}$',r'$\delta_{32}$',r'$\delta_{33}$',r'$\delta_{34}$',r'$\delta_{35}$',r'$\delta_{36}$',r'$\delta_{37}$',r'$\delta_{38}$',r'$\delta_{39}$',r'$\delta_{40}$',r'$\delta_{41}$',r'$\delta_{42}$',r'$\delta_{43}$',r'$\delta_{44}$',r'$\delta_{45}$',r'$\delta_{46}$',r'$\delta_{47}$',r'$\delta_{48}$',r'$\delta_{49}$',r'$\delta_{50}$',r'$\delta_{51}$',r'$\delta_{52}$',r'$\delta_{53}$',r'$\delta_{54}$',r'$\delta_{55}$',r'$\delta_{56}$',r'$\delta_{57}$',r'$\delta_{58}$',r'$\delta_{59}$',r'$\delta_{60}$',r'$\delta_{61}$',r'$\delta_{62}$',r'$\delta_{63}$',r'$\delta_{64}$',r'$\delta_{65}$',r'$\delta_{66}$',r'$\delta_{67}$',r'$\delta_{68}$',r'$\delta_{69}$',r'$\delta_{70}$',r'$\delta_{71}$',r'$\delta_{72}$',r'$\delta_{73}$',r'$\delta_{74}$',r'$\delta_{75}$',r'$\delta_{76}$',r'$\delta_{77}$',r'$\delta_{78}$',r'$\delta_{79}$',r'$\delta_{80}$',r'$\delta_{81}$',r'$\delta_{82}$',r'$\delta_{83}$',r'$\delta_{84}$',r'$\delta_{85}$',r'$\delta_{86}$',r'$\delta_{87}$',r'$\delta_{88}$',r'$\delta_{89}$',r'$\delta_{90}$',r'$\delta_{91}$',r'$\delta_{92}$',r'$\delta_{93}$',r'$\delta_{94}$']
    plt.rcParams.update({'font.size': 22})
    labels_3=list( labels[i] for i in selection_list )
    labels=labels_3
    ndims=np.shape(target_samples)[1]
    print(np.shape(target_samples)[0])
    red_bins=30
    density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
   
    blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    blue_bins=blue_bins.astype(int).tolist()

 
    figure=corner.corner(target_samples,color='red',bins=red_bins)#,labels=labels,max_n_ticks=2) #,levels=(0.68,95.,99.7))
    
    corner.corner(nf_samples,color='blue',bins=blue_bins,fig=figure)  # , levels=(0.68,95.,99.7))
    
    ndims=np.shape(nf_samples)[1]
    
    #figure=DrawCIs(figure,hdpis_dict_true,hdpis_dict_nf,ndims,selection_list,labels)
    #figure.tight_layout()
    
    red_line = mlines.Line2D([], [],lw=30, color='red', label='true')
    blue_line = mlines.Line2D([], [],lw=30, color='blue', label='pred')
    hdpi1_line = mlines.Line2D([], [],lw=5, color='black',linestyle='-', label='HPDI$_{1\sigma}$')
    hdpi2_line = mlines.Line2D([], [],lw=5, color='black',linestyle='--', label='HPDI$_{2\sigma}$')
    hdpi3_line = mlines.Line2D([], [],lw=5, color='black',linestyle='-.', label='HPDI$_{3\sigma}$')
    
    
    plt.legend(handles=[blue_line,red_line,hdpi1_line,hdpi2_line,hdpi3_line],bbox_to_anchor=(-ndims+9.8, ndims+.5, 1., 0.) ,fontsize='xx-large')
    
    figure.set_size_inches(37,37)
    plt.tight_layout()
    plt.savefig(path_to_plot,pil_kwargs={'quality':60})
    
    plt.close()


    return

path_to_results='./results/newmet_1/run_2_b/'
#results=os.listdir(path_to_results)
#nsamples_test=100002
nf_sample_exists=True



labels=[r'$\mu$',r'$\delta_{1}$',r'$\delta_{2}$',r'$\delta_{3}$',r'$\delta_{4}$',r'$\delta_{5}$',r'$\delta_{6}$',r'$\delta_{7}$',r'$\delta_{8}$',r'$\delta_{9}$',r'$\delta_{10}$',r'$\delta_{11}$',r'$\delta_{12}$',r'$\delta_{13}$',r'$\delta_{14}$',r'$\delta_{15}$',r'$\delta_{16}$',r'$\delta_{17}$',r'$\delta_{18}$',r'$\delta_{19}$',r'$\delta_{20}$',r'$\delta_{21}$',r'$\delta_{22}$',r'$\delta_{23}$',r'$\delta_{24}$',r'$\delta_{25}$',r'$\delta_{26}$',r'$\delta_{27}$',r'$\delta_{28}$',r'$\delta_{29}$',r'$\delta_{30}$',r'$\delta_{31}$',r'$\delta_{32}$',r'$\delta_{33}$',r'$\delta_{34}$',r'$\delta_{35}$',r'$\delta_{36}$',r'$\delta_{37}$',r'$\delta_{38}$',r'$\delta_{39}$',r'$\delta_{40}$',r'$\delta_{41}$',r'$\delta_{42}$',r'$\delta_{43}$',r'$\delta_{44}$',r'$\delta_{45}$',r'$\delta_{46}$',r'$\delta_{47}$',r'$\delta_{48}$',r'$\delta_{49}$',r'$\delta_{50}$',r'$\delta_{51}$',r'$\delta_{52}$',r'$\delta_{53}$',r'$\delta_{54}$',r'$\delta_{55}$',r'$\delta_{56}$',r'$\delta_{57}$',r'$\delta_{58}$',r'$\delta_{59}$',r'$\delta_{60}$',r'$\delta_{61}$',r'$\delta_{62}$',r'$\delta_{63}$',r'$\delta_{64}$',r'$\delta_{65}$',r'$\delta_{66}$',r'$\delta_{67}$',r'$\delta_{68}$',r'$\delta_{69}$',r'$\delta_{70}$',r'$\delta_{71}$',r'$\delta_{72}$',r'$\delta_{73}$',r'$\delta_{74}$',r'$\delta_{75}$',r'$\delta_{76}$',r'$\delta_{77}$',r'$\delta_{78}$',r'$\delta_{79}$',r'$\delta_{80}$',r'$\delta_{81}$',r'$\delta_{82}$',r'$\delta_{83}$',r'$\delta_{84}$',r'$\delta_{85}$',r'$\delta_{86}$',r'$\delta_{87}$',r'$\delta_{88}$',r'$\delta_{89}$',r'$\delta_{90}$',r'$\delta_{91}$',r'$\delta_{92}$',r'$\delta_{93}$',r'$\delta_{94}$']

#results_dict={'ndims':[],'nbijectors':[],'bijector':[],'nsamples':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'kl_divergence':[],'ks_test_mean':[],'ks_test_median':[],'Wasserstein_median':[],'Wasserstein_mean':[],'frob_norm':[],'hidden_layers':[],'batch_size':[],'time':[]}


results_dict_newmetrics={'KS_pv_mean':[],'KS_pv_std':[],'KS_st_mean':[],'KS_st_std':[],'SWD_mean':[],'SWD_std':[]}




###retrain
#epochs=1000
#history,nf_dist=retrain_model(model,X_data_train,epochs,batch_size)
#saver_2(nf_dist,path_to_results,iter)
#saver(nf_dist,path_to_results)
#t_losses=history.history['loss']
#v_losses=history.history['val_loss']



files=os.listdir(path_to_results)

    
ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser=load_hyperparams(path_to_results)




#targ_dist=MixtureGaussian(ndims)


X_data_train_file = '../../NFLikelihoods-2/Toy-Likelihood/data/X_data_LF100_2M'
X_data_test_file = '../../NFLikelihoods-2/Toy-Likelihood/data/X_data_test_LF100_500000'
logprobs_data_test_file = '../../NFLikelihoods-2/Toy-Likelihood/data/logprobs_data_test_LF100_500000'

X_data_train,X_data_test,logprobs_data_test=loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file)

preporcess_params_path='preprocess_data_toyllhdfit.pcl'
preprocess_params=load_preprocess_params(preporcess_params_path)

#n_total_test_samples=nsamples_test
#X_data_train=X_data_train[:n_total_test_samples,:]

#X_data_test=targ_dist.sample(nsamples_test).numpy()
    
    


nf_dist=create_flow(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser)
nf_dist,model=load_model(nf_dist,path_to_results,ndims,lr=.00001)
files=os.listdir(path_to_results)
        
        
#nf_dist=SoftClipTransform(nf_dist,1e-4)
        
        
        
        
nsamples_test=200000
save_sample(nf_dist,path_to_results,sample_size=nsamples_test,iter_size=400)

nf_sample=load_sample(path_to_results)


nf_sample=load_sample(path_to_results)

nf_sample=postprocess_data(nf_sample,preprocess_params)

n_total_test_samples=np.shape(nf_sample)[0]
n_total_test_samples=200000
X_data_test=X_data_test[:n_total_test_samples,:]
nf_sample=nf_sample[:n_total_test_samples,:]

dist_1 = tf.cast(nf_sample,tf.float64)
dist_2 = tf.cast(X_data_test,tf.float64)
        
TwoSampleTestInputs_tf = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                      dist_2_input = dist_2,
                                                      niter = 100,
                                                      batch_size = int(n_total_test_samples/100),
                                                      dtype_input = tf.float64,
                                                      seed_input = 0,
                                                      use_tf = True,
                                                      verbose = True)
                                                      
                                                      
KSTest_tf = GMetrics.KSTest(TwoSampleTestInputs_tf,
                                                                progress_bar = True,
                                                                verbose = True)
SWDMetric_tf = GMetrics.SWDMetric(TwoSampleTestInputs_tf,
                                                                progress_bar = True,
                                                                verbose = True)

                                                                                                 
                                                            
KSTest_tf.Test_np()

KSres =KSTest_tf.Results[-1].result_value
print("Keys:", list(KSres.keys()))
print("Value dtype:", [type(x) for x in list(KSres.values())])
print("Value shape:", [x.shape for x in list(KSres.values())])

print('Result:',KSres.get('pvalue_means'))
                                                      
                                                      
SaveNewMetrics(KSres,'KS',path_to_results)

        
KS_pvalues=GetReusltsForPOI(KSres.get('pvalue_lists'),labels,path_to_results,'KS_pv')

print(KS_pvalues[14:21])

GetReusltsForPOI(KSres.get('statistic_lists'),labels,path_to_results,'KS_st')
        
        
     
KS_pv_mean,KS_pv_std=MeansAndStdMetrics(KSres.get('pvalue_means'))
                                                                
KS_st_mean,KS_st_std=MeansAndStdMetrics(KSres.get('statistic_means'))
                                                                
                                                                
print('KS mean', KS_pv_mean)
print('KS std', KS_pv_std)
                                                            
SWDMetric_tf.Test_np()

SWDres =SWDMetric_tf.Results[-1].result_value
print("Keys:", list(SWDres.keys()))
print("Value dtype:", [type(x) for x in list(SWDres.values())])
print("Value shape:", [x.shape for x in list(SWDres.values())])

print('Result:',SWDres.get('metric_means'))
                                                      
                                                      
SaveNewMetrics(SWDres,'SWD',path_to_results)
SWDmean,SWDstd=MeansAndStdMetrics(SWDres.get('metric_means'))
print('hey SWD')
print(SWDres.get('metric_lists'))
print(np.shape(SWDres.get('metric_lists')))
     

print('SWD means', SWDmean)
print('SWD stds', SWDstd)
                                                            
print('####DONE WITH NEW TESTS#####')
       
results_dict_newmetrics=ResultsToDict_NewMetrics(KS_pv_mean,KS_pv_std,KS_st_mean,KS_st_std,SWDmean,SWDstd)
                                                        
SaveResultsCurrentNewMetrics(results_dict_newmetrics,path_to_results)
        
        
corner_plot_name='corner_lot_newmetrics.png'
marginal_plot_name='marginal_plot_newmetrics.png'
results_name='new_metrics_'
PlotsandHDPI.ResultstoFrame(path_to_results,results_dict_newmetrics,n_total_test_samples,results_name)
PlotsandHDPI.GeneratePlotandHDPIResults(X_data_test,nf_sample,path_to_results,ndims,n_total_test_samples, results_name,KS_pvalues,corner_plot_name,marginal_plot_name)
selection_list=[0,86,87,88,89,90,91,92,93,94]
#selection_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,10,11,38,39]
#selection_list=[14,15]
path_to_plot=path_to_results+'/cornerplot_test_all_'+results_name+'_'+str(n_total_test_samples)+'.png'
CornerPlotter(X_data_test[:100000,selection_list],nf_sample[:100000,selection_list],path_to_plot,selection_list)

       
       

        
