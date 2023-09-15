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
#import Distributions,Bijectors
#import Trainer_2 as Trainer
#import Metrics as Metrics
from statistics import mean,median
import pickle
import  matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer
import os
import math
import sys
sys.path.append('../code')
import GenerativeModelsMetrics as GMetrics
#import CorrelatedGaussians
#import Plotters
#import MixtureDistributions
#import compare_logprobs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import PlotsandHDPIforReload as PlotsandHDPI
import matplotlib.lines as mlines

import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils

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


def MixtureGaussian(ndims):

        if ndims==4:
            targ_dist=MixtureDistributions.MixGauss4()
        
        if ndims==8:
            targ_dist=MixtureDistributions.MixGauss8()

        if ndims==16:
            targ_dist=MixtureDistributions.MixGauss16()

        if ndims==32:
            targ_dist=MixtureDistributions.MixGauss32()
        
        
        if ndims==64:
            targ_dist=MixtureDistributions.MixGauss64()
    
        if ndims==100:
            targ_dist=MixtureDistributions.MixGauss100()
            
        return targ_dist

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
        bijector=Bijectors.MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser,perm_style='reverse')
        


    if bijector_name=='MAFN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser)
        
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


    #print(np.shape(sample_all))
    with open(path_to_results+'/nf_sample.npy', 'wb') as f:
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

    #print(pd.DataFrame(test_means))

    return


def GetReusltsForPOI(results_list,labels,path_to_results,name):

    results_list=np.transpose(results_list)

    
    POI_results_mean=np.mean(results_list,axis=1)

    POI_results_mean_frame=pd.DataFrame(POI_results_mean).transpose()
    POI_results_mean_frame.columns =labels
    POI_results_mean_frame.to_csv(path_to_results+'/'+name+'_POI_results_sf.txt',index=False)

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

    max_conditions=[5.6119569040384665, 9.408321669056832, 5.921143923401007, 2.0987539948122715, 1.7335918107131771, 1.5496947863239248, 6.015819026904569, 5.357000788249108, 5.078956527236209, 1.9706613161898978, 1.9101265446366096, 1.6733549612022363, 5.136582353186947, 5.450362313377298, 2.0646370061806674, 1.9235362337821624, 1.7261000227703496, 1.7584699885882422, 1.7261145162920541, 1.734678341121733, 5.303146000538385, 1.2408237363374655, 4.5208285276014655, 4.611096424211525, 4.67247424059945, 1.4438519838181176, 1.7425589680115732, 1.6612583351965835, 1.7815774951237893, 1.738228427666701, 4.750417146123906, 4.772144294967443, 4.098516986699341, 4.687790767222508, 4.648232436632133, 4.73335050833763, 4.992481060191991, 4.5685777884876435, 4.692557719410474, 4.6848372177608635, 5.217863521870644, 4.652724547849794, 4.613649584431962, 4.648134663110919, 4.982229325918564, 4.422509818819, 4.439536968846636, 4.807716213331392, 4.914197213991386, 5.085327218848985, 5.209548924005082, 4.733598946040475, 4.439572029870957, 5.159573608150739, 4.434700977430681, 4.840175450673164, 5.392032812553973, 5.141916761568047, 5.011726785228644, 4.670226661815609, 4.955455036694929, 5.042015303366961, 4.412874228275154, 4.710184677756211, 4.912336401667579, 4.819064018516058, 4.416760141828074, 5.172366091120923, 4.670204816478531, 5.017447935350553, 4.888395493928422, 4.587133254342274, 4.964054682574907, 4.766753149198907, 4.384927585158509, 4.915176688322467, 4.520501373940944, 2.9103100032749665, 5.586215137792808, 3.2856681007051116, 3.4139782803045717, 2.147317265602916, 2.260629889692233, 3.477996125332174, 4.904049555630162, 2.7418604159447146, 3.9871097614145503, 2.123393534531489, 2.1199681188827193]


    min_conditions=[-1.5084377926591506, -1.1403203148728418, -2.496500952464059, -3.4063185703305665, -1.8244945529911047, -1.7465928259936245, -1.1664721288961046, -1.4121997300790023, -1.3784733701350902, -2.043872394062949, -2.415854255775898, -1.975487952321519, -1.6098315102808707, -1.3127420122099642, -3.2462309463922785, -2.1201205998294044, -1.7534069948736217, -1.7530192327414973, -1.7105445832971058, -1.7308037311615643, -1.1121057269957326, -2.1710137373887326, -4.6875861871041, -4.2721357126903365, -4.536547590491557, -2.6782964764941752, -1.7161104528406887, -1.7718911017041539, -1.6826392859341341, -1.7312847793305364, -4.335857116055725, -4.439620071795575, -3.66228858986652, -4.770938296719519, -4.54863749293239, -4.993337476228878, -4.835147802468657, -4.782566882850699, -4.600584766861787, -4.708251144639543, -4.5494115434656806, -4.686817053828232, -4.704545726595088, -5.190751195092813, -4.520832661523522, -4.537798035713625, -4.284177009139489, -4.681677266614382, -4.565935528255109, -4.7353084066141795, -4.706371837702957, -4.556178308190562, -4.78790250770006, -4.843064350859609, -4.669400658073029, -4.638740107100772, -4.398275377744491, -4.8102190132314835, -4.949473271379748, -4.669071920330874, -5.485755070769344, -4.693172065282441, -4.460176876493423, -4.848101524340766, -4.279124191441311, -4.828178614796366, -4.420692411259104, -4.8010547556168435, -4.325342737868171, -4.816959489897645, -5.072689643860321, -4.170787843285713, -4.342952438293287, -4.762433021903761, -4.376852083237983, -4.531748269282737, -4.998387076297546, -1.9253108978562523, -4.358093562823174, -4.87016585330077, -3.5672813718258976, -2.1257475777616324, -1.9168860977284283, -4.240789174065665, -4.7142165879651134, -1.990144028197836, -4.239242098184308, -2.129705565636902, -1.9856780689375653]
 
    bijector=tfb.SoftClip(low=min_conditions, high=max_conditions,hinge_softness=hinge)

    nf_dist=tfd.TransformedDistribution(nf_dist,bijector)

    return nf_dist
def CornerPlotter(target_samples,nf_samples,path_to_plot,selection_list):


    labels = [r"$|h_{0}^{(0)}|$",
                   r"$|h_{+}^{(0)}|$",
                   r"$|h_{-}^{(0)}|$",
                   r"$\arg\left(h_{0}^{(0)}\right)$",
                   r"$\arg\left(h_{+}^{(0)}\right)$",
                   r"$\arg\left(h_{-}^{(0)}\right)$",
                   r"$|h_{0}^{(1)}|$",
                   r"$|h_{+}^{(1)}|$",
                   r"$|h_{-}^{(1)}|$",
                   r"$\arg\left(h_{0}^{(1)}\right)$",
                   r"$\arg\left(h_{+}^{(1)}\right)$",
                   r"$\arg\left(h_{-}^{(1)}\right)$",
                   r"$|h_{+}^{(2)}|$",
                   r"$|h_{-}^{(2)}|$",
                   r"$\arg\left(h_{+}^{(2)}\right)$",
                   r"$\arg\left(h_{-}^{(2)}\right)$",
                   r"$|h_{0}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{0}^{(\mathrm{MP})}\right)$",
                   r"$|h_{1}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{1}^{(\mathrm{MP})}\right)$",
                   r"$|h_{2}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{1}^{(\mathrm{MP})}\right)$",
                   r"$F_{Bs}/F_{Bd}$",
                   r"$\delta Gs_{Gs}$",
                   r"$F_{Bs}$",
                   r"$\lambda_{8}$",
                   r"$\alpha_{1}(K^{*})$",
                   r"$\alpha_{2}(K^{*})$",
                   r"$\alpha_{2}(\phi)$",
                   r"$\alpha_{1}(K^{'})$",
                   r"$a_{0}^{A0}$",
                   r"$a_{0}^{A1}$",
                   r"$a_{0}^{T1}$",
                   r"$a_{0}^{T23}$",
                   r"$a_{0}^{V}$",
                   r"$a_{1}^{A0}$",
                   r"$a_{1}^{A1}$",
                   r"$a_{1}^{A12}$",
                   r"$a_{1}^{T1}$",
                   r"$a_{1}^{T2}$",
                   r"$a_{1}^{T23}$",
                   r"$a_{1}^{V}$",
                   r"$a_{2}^{A0}$",
                   r"$a_{2}^{A1}$",
                   r"$a_{2}^{A12}$",
                   r"$a_{2}^{T1}$",
                   r"$a_{2}^{T2}$",
                   r"$a_{2}^{T23}$",
                   r"$a_{2}^{V}$",
                   r"$a_{0\\,\\phi}^{A0}$",
                   r"$a_{0\,\phi}^{A1}$",
                   r"$a_{0\,\phi}^{T1}$",
                   r"$a_{0\,\phi}^{T23}$",
                   r"$a_{0\,\phi}^{V}$",
                   r"$a_{1\,\phi}^{A0}$",
                   r"$a_{1\,\phi}^{A1}$",
                   r"$a_{1\,\phi}^{A12}$",
                   r"$a_{1\,\phi}^{T1}$",
                   r"$a_{1\,\phi}^{T2}$",
                   r"$a_{1\,\phi}^{T23}$",
                   r"$a_{1\,\phi}^{V}$",
                   r"$a_{2\,\phi}^{A0}$",
                   r"$a_{2\,\phi}^{A1}$",
                   r"$a_{2\,\phi}^{A12}$",
                   r"$a_{2\,\phi}^{T1}$",
                   r"$a_{2\,\phi}^{T2}$",
                   r"$a_{2\,\phi}^{T23}$",
                   r"$a_{2\,\phi}^{V}$",
                   r"$b_{0\,f_{0}}$",
                   r"$b_{0\,f_{T}}$",
                   r"$b_{0\,f_{+}}$",
                   r"$b_{1\,f_{0}}$",
                   r"$b_{1\,f_{T}}$",
                   r"$b_{1\,f_{+}}$",
                   r"$b_{2\,f_{0}}$",
                   r"$b_{2\,f_{T}}$",
                   r"$b_{2\,f_{+}}$",
                   r"$c^{LQ\,1}_{1123}$",
                   r"$c^{LQ\,1}_{2223}$",
                   r"$c^{Ld}_{1123}$",
                   r"$c^{Ld}_{2223}$",
                   r"$c^{LedQ}_{11}$",
                   r"$c^{LedQ}_{22}$",
                   r"$c^{Qe}_{2311}$",
                   r"$c^{Qe}_{2322}$",
                   r"$c^{ed}_{1123}$",
                   r"$c^{ed}_{2223}$",
                   r"$c^{\prime\,LedQ}_{11}$",
                   r"$c^{\prime\,LedQ}_{22}$",
                 
                 ]
    plt.rcParams.update({'font.size': 22})
    labels_3=list( labels[i] for i in selection_list )
    labels=labels_3
    ndims=np.shape(target_samples)[1]

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

path_to_results='results/official_results/'
#results=os.listdir(path_to_results)
#nsamples_test=100002
nf_sample_exists=True


labels = [r"$|h_{0}^{(0)}|$",
                   r"$|h_{+}^{(0)}|$",
                   r"$|h_{-}^{(0)}|$",
                   r"$\arg\left(h_{0}^{(0)}\right)$",
                   r"$\arg\left(h_{+}^{(0)}\right)$",
                   r"$\arg\left(h_{-}^{(0)}\right)$",
                   r"$|h_{0}^{(1)}|$",
                   r"$|h_{+}^{(1)}|$",
                   r"$|h_{-}^{(1)}|$",
                   r"$\arg\left(h_{0}^{(1)}\right)$",
                   r"$\arg\left(h_{+}^{(1)}\right)$",
                   r"$\arg\left(h_{-}^{(1)}\right)$",
                   r"$|h_{+}^{(2)}|$",
                   r"$|h_{-}^{(2)}|$",
                   r"$\arg\left(h_{+}^{(2)}\right)$",
                   r"$\arg\left(h_{-}^{(2)}\right)$",
                   r"$|h_{0}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{0}^{(\mathrm{MP})}\right)$",
                   r"$|h_{1}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{1}^{(\mathrm{MP})}\right)$",
                   r"$|h_{2}^{(\mathrm{MP})}|$",
                   r"$\arg\left(h_{1}^{(\mathrm{MP})}\right)$",
                   r"$F_{Bs}/F_{Bd}$",
                   r"$\delta Gs_{Gs}$",
                   r"$F_{Bs}$",
                   r"$\lambda_{8}$",
                   r"$\alpha_{1}(K^{*})$",
                   r"$\alpha_{2}(K^{*})$",
                   r"$\alpha_{2}(\phi)$",
                   r"$\alpha_{1}(K^{'})$",
                   r"$a_{0}^{A0}$",
                   r"$a_{0}^{A1}$",
                   r"$a_{0}^{T1}$",
                   r"$a_{0}^{T23}$",
                   r"$a_{0}^{V}$",
                   r"$a_{1}^{A0}$",
                   r"$a_{1}^{A1}$",
                   r"$a_{1}^{A12}$",
                   r"$a_{1}^{T1}$",
                   r"$a_{1}^{T2}$",
                   r"$a_{1}^{T23}$",
                   r"$a_{1}^{V}$",
                   r"$a_{2}^{A0}$",
                   r"$a_{2}^{A1}$",
                   r"$a_{2}^{A12}$",
                   r"$a_{2}^{T1}$",
                   r"$a_{2}^{T2}$",
                   r"$a_{2}^{T23}$",
                   r"$a_{2}^{V}$",
                   r"$a_{0\\,\\phi}^{A0}$",
                   r"$a_{0\,\phi}^{A1}$",
                   r"$a_{0\,\phi}^{T1}$",
                   r"$a_{0\,\phi}^{T23}$",
                   r"$a_{0\,\phi}^{V}$",
                   r"$a_{1\,\phi}^{A0}$",
                   r"$a_{1\,\phi}^{A1}$",
                   r"$a_{1\,\phi}^{A12}$",
                   r"$a_{1\,\phi}^{T1}$",
                   r"$a_{1\,\phi}^{T2}$",
                   r"$a_{1\,\phi}^{T23}$",
                   r"$a_{1\,\phi}^{V}$",
                   r"$a_{2\,\phi}^{A0}$",
                   r"$a_{2\,\phi}^{A1}$",
                   r"$a_{2\,\phi}^{A12}$",
                   r"$a_{2\,\phi}^{T1}$",
                   r"$a_{2\,\phi}^{T2}$",
                   r"$a_{2\,\phi}^{T23}$",
                   r"$a_{2\,\phi}^{V}$",
                   r"$b_{0\,f_{0}}$",
                   r"$b_{0\,f_{T}}$",
                   r"$b_{0\,f_{+}}$",
                   r"$b_{1\,f_{0}}$",
                   r"$b_{1\,f_{T}}$",
                   r"$b_{1\,f_{+}}$",
                   r"$b_{2\,f_{0}}$",
                   r"$b_{2\,f_{T}}$",
                   r"$b_{2\,f_{+}}$",
                   r"$c^{LQ\,1}_{1123}$",
                   r"$c^{LQ\,1}_{2223}$",
                   r"$c^{Ld}_{1123}$",
                   r"$c^{Ld}_{2223}$",
                   r"$c^{LedQ}_{11}$",
                   r"$c^{LedQ}_{22}$",
                   r"$c^{Qe}_{2311}$",
                   r"$c^{Qe}_{2322}$",
                   r"$c^{ed}_{1123}$",
                   r"$c^{ed}_{2223}$",
                   r"$c^{\prime\,LedQ}_{11}$",
                   r"$c^{\prime\,LedQ}_{22}$",
                 
                 ]

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


X_data_train_file = 'data/X_data_train_flavor_new_1p5M'
X_data_test_file = 'data/X_data_test_flavor_new_500K'
logprobs_data_test_file = 'data/X_data_test_logprob_flavor_new_500k'

X_data_train,X_data_test,logprobs_data_test=loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file)

preporcess_params_path='preprocess_data_flavorfit_new.pcl'
preprocess_params=load_preprocess_params(preporcess_params_path)

#n_total_test_samples=nsamples_test
#X_data_train=X_data_train[:n_total_test_samples,:]

#X_data_test=targ_dist.sample(nsamples_test).numpy()
    
    


nf_dist=create_flow(bijector_name,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser)
nf_dist,model=load_model(nf_dist,path_to_results,ndims,lr=.00001)
files=os.listdir(path_to_results)
        
        
nf_dist=SoftClipTransform(nf_dist,1e-4)
        
        
        
        
nsamples_test=2000
save_sample(nf_dist,path_to_results,sample_size=nsamples_test,iter_size=400)

nf_sample=load_sample(path_to_results)

#nf_sample=load_sample(path_to_results)

nf_sample=postprocess_data(nf_sample,preprocess_params)

n_total_test_samples=np.shape(nf_sample)[0]
n_total_test_samples=np.shape(nf_sample)[0]

#exit()
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



                                                      
                                                      
SaveNewMetrics(KSres,'KS',path_to_results)

        
KS_pvalues=GetReusltsForPOI(KSres.get('pvalue_lists'),labels,path_to_results,'KS_pv')



GetReusltsForPOI(KSres.get('statistic_lists'),labels,path_to_results,'KS_st')
        
        
     
KS_pv_mean,KS_pv_std=MeansAndStdMetrics(KSres.get('pvalue_means'))
                                                                
KS_st_mean,KS_st_std=MeansAndStdMetrics(KSres.get('statistic_means'))
                                                                
                                                                
print('KS mean', KS_pv_mean)
print('KS std', KS_pv_std)
                                                            
SWDMetric_tf.Test_np()

SWDres =SWDMetric_tf.Results[-1].result_value



                                                      
                                                      
SaveNewMetrics(SWDres,'SWD',path_to_results)
SWDmean,SWDstd=MeansAndStdMetrics(SWDres.get('metric_means'))

     

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
selection_list=[77,78,79,80,81,82,83,84,85,86,87,88]
#selection_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,10,11,38,39]
#selection_list=[14,15]
path_to_plot=path_to_results+'/cornerplot_test_all_'+results_name+'_'+str(n_total_test_samples)+'.png'
CornerPlotter(X_data_test[:100000,selection_list],nf_sample[:100000,selection_list],path_to_plot,selection_list)

       
       

        

        
'''
        #kl_divergence,ks_median,ks_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr=ComputeMetrics(X_data_test,nf_sample)
                                    
        #results_dict=ResultsToDict(results_dict,ndims,bijector_name,nbijectors,nsamples,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,w_distance_median,w_distance_mean,frob_norm,hllabel,batch_size,eps_regulariser,regulariser)


  
        corner_start=timer()
        
        exit()
        Plotters.marginal_plot(X_data_test,nf_sample,path_to_results,ndims)
        try:
            Plotters.cornerplotter(X_data_test,nf_sample,path_to_results,ndims,norm=True)
        except:
            print('no corner plot')
        corner_end=timer()
        print(corner_end-corner_start)
        tf.keras.backend.clear_session()
#except:
#    print('no corner plot possible')
'''
