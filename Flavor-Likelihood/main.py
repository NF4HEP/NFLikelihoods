import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
import pandas as pd
import pickle
from timeit import default_timer as timer
import traceback
from typing import Dict, Any
import PlotsandHDPI


sys.path.append('../code')
import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
'''

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

def preprocess_data(data,preprocess_params):
    
    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    preprocess_data=(data-means)/stds
    
    return preprocess_data


def postprocess_data(data,preprocess_params):

    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    postprocess_data=data*stds+means
    
    return postprocess_data


def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    return preprocess_params

def SoftClipTransform(nf_dist,hinge):

    max_conditions=[5.6119569040384665, 9.408321669056832, 5.921143923401007, 2.0987539948122715, 1.7335918107131771, 1.5496947863239248, 6.015819026904569, 5.357000788249108, 5.078956527236209, 1.9706613161898978, 1.9101265446366096, 1.6733549612022363, 5.136582353186947, 5.450362313377298, 2.0646370061806674, 1.9235362337821624, 1.7261000227703496, 1.7584699885882422, 1.7261145162920541, 1.734678341121733, 5.303146000538385, 1.2408237363374655, 4.5208285276014655, 4.611096424211525, 4.67247424059945, 1.4438519838181176, 1.7425589680115732, 1.6612583351965835, 1.7815774951237893, 1.738228427666701, 4.750417146123906, 4.772144294967443, 4.098516986699341, 4.687790767222508, 4.648232436632133, 4.73335050833763, 4.992481060191991, 4.5685777884876435, 4.692557719410474, 4.6848372177608635, 5.217863521870644, 4.652724547849794, 4.613649584431962, 4.648134663110919, 4.982229325918564, 4.422509818819, 4.439536968846636, 4.807716213331392, 4.914197213991386, 5.085327218848985, 5.209548924005082, 4.733598946040475, 4.439572029870957, 5.159573608150739, 4.434700977430681, 4.840175450673164, 5.392032812553973, 5.141916761568047, 5.011726785228644, 4.670226661815609, 4.955455036694929, 5.042015303366961, 4.412874228275154, 4.710184677756211, 4.912336401667579, 4.819064018516058, 4.416760141828074, 5.172366091120923, 4.670204816478531, 5.017447935350553, 4.888395493928422, 4.587133254342274, 4.964054682574907, 4.766753149198907, 4.384927585158509, 4.915176688322467, 4.520501373940944, 2.9103100032749665, 5.586215137792808, 3.2856681007051116, 3.4139782803045717, 2.147317265602916, 2.260629889692233, 3.477996125332174, 4.904049555630162, 2.7418604159447146, 3.9871097614145503, 2.123393534531489, 2.1199681188827193]
    
    
    min_conditions=[-1.5084377926591506, -1.1403203148728418, -2.496500952464059, -3.4063185703305665, -1.8244945529911047, -1.7465928259936245, -1.1664721288961046, -1.4121997300790023, -1.3784733701350902, -2.043872394062949, -2.415854255775898, -1.975487952321519, -1.6098315102808707, -1.3127420122099642, -3.2462309463922785, -2.1201205998294044, -1.7534069948736217, -1.7530192327414973, -1.7105445832971058, -1.7308037311615643, -1.1121057269957326, -2.1710137373887326, -4.6875861871041, -4.2721357126903365, -4.536547590491557, -2.6782964764941752, -1.7161104528406887, -1.7718911017041539, -1.6826392859341341, -1.7312847793305364, -4.335857116055725, -4.439620071795575, -3.66228858986652, -4.770938296719519, -4.54863749293239, -4.993337476228878, -4.835147802468657, -4.782566882850699, -4.600584766861787, -4.708251144639543, -4.5494115434656806, -4.686817053828232, -4.704545726595088, -5.190751195092813, -4.520832661523522, -4.537798035713625, -4.284177009139489, -4.681677266614382, -4.565935528255109, -4.7353084066141795, -4.706371837702957, -4.556178308190562, -4.78790250770006, -4.843064350859609, -4.669400658073029, -4.638740107100772, -4.398275377744491, -4.8102190132314835, -4.949473271379748, -4.669071920330874, -5.485755070769344, -4.693172065282441, -4.460176876493423, -4.848101524340766, -4.279124191441311, -4.828178614796366, -4.420692411259104, -4.8010547556168435, -4.325342737868171, -4.816959489897645, -5.072689643860321, -4.170787843285713, -4.342952438293287, -4.762433021903761, -4.376852083237983, -4.531748269282737, -4.998387076297546, -1.9253108978562523, -4.358093562823174, -4.87016585330077, -3.5672813718258976, -2.1257475777616324, -1.9168860977284283, -4.240789174065665, -4.7142165879651134, -1.990144028197836, -4.239242098184308, -2.129705565636902, -1.9856780689375653]
 
    bijector=tfb.SoftClip(low=min_conditions, high=max_conditions,hinge_softness=hinge)

    nf_dist=tfd.TransformedDistribution(nf_dist,bijector)

    return nf_dist

####target dist
X_data_train_file = 'data/X_data_train_flavor_new_1p5M'
X_data_test_file = 'data/X_data_test_flavor_new_500K'
logprobs_data_test_file = 'data/X_data_test_logprob_flavor_new_500k'

X_data_train_all,X_data_test,logprobs_data_test=loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file)

hinge=1e-4


preporcess_params_path='preprocess_data_flavorfit_new.pcl'
preprocess_params=load_preprocess_params(preporcess_params_path)
X_data_train_prep_all=preprocess_data(X_data_train_all,preprocess_params)

ndims=np.shape(X_data_test)[1]
### Initialize hyperparameters lists ###
ndims_list=[ndims]
corr_uncorr_list=["corr"]
regulariser_list=[None]
eps_regularisers=[0]
nsamples_list=[500000]
batch_size_list=[512]
bijectors_list=['MsplineN']
###type of permutiation between flow layers
perm_style='reverse'
activation_list=['relu']
nbijectors_list=[2]
hidden_layers_list=[[128,128,128]]
seeds_list = [None]
n_displays=1

### Initialize variables for the neural splines ###
range_min_list=[-5]
spline_knots_list=[8]

### Initialize train hyerparameters ###
ntest_samples=5000
epochs=5
lr_orig=.001
patience=50
min_delta_patience=.0001
lr_change=.2
seed_dist = None
seed_test = None

X_data_test=X_data_test[:ntest_samples,:]
### Initialize output dir ###
mother_output_dir='results/test_newmetrics_1/'
try:
    os.mkdir(mother_output_dir)
except:
    print('file exists')

### Initialize dictionaries ###


results_dict: Dict[str,Any] = {'run_n': [],'run_seed': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'activation':[],'spline_knots':[],'range_min':[],'eps_regulariser':[],'regulariser':[],'hidden_layers':[],'batch_size':[],'epochs_input':[],'epochs_output':[],'time':[],'KS_pv_mean':[],'KS_pv_std':[],'KS_st_mean':[],'KS_st_std':[],'SWD_mean':[],'SWD_std':[]}

hyperparams_dict: Dict[str,Any] = {'run_n': [],'run_seed': [], 'ndims':[],'nsamples':[],'correlation':[],'nbijectors':[],'bijector':[],'spline_knots':[],'range_min':[],'hidden_layers':[],'batch_size':[],'activation':[],'eps_regulariser':[],'regulariser':[],'dist_seed':[],'test_seed':[]}

### Create 'log' file ####
log_file_name = Utils.create_log_file(mother_output_dir,results_dict)

### Run loop  ###
run_number = 2
n_runs = len(ndims_list)*len(seeds_list)*len(nsamples_list)*len(corr_uncorr_list)*len(activation_list)*len(eps_regularisers)*len(regulariser_list)*len(bijectors_list)*len(nbijectors_list)*len(spline_knots_list)*len(range_min_list)*len(batch_size_list)*len(hidden_layers_list)
for ndims in ndims_list:

    for seed in seeds_list:
        for nsamples in nsamples_list:
            for activation in activation_list:
                for eps_regulariser in eps_regularisers:
                    for regulariser in regulariser_list:
                        for bijector_name in bijectors_list:
                            for nbijectors in nbijectors_list:
                                for spline_knots in spline_knots_list:
                                    for range_min in range_min_list:
                                        for batch_size in batch_size_list:
                                            for hidden_layers in hidden_layers_list:
                                                for corr in corr_uncorr_list:
                                                    #Utils.reset_random_seeds(seed)
                                                    run_number = run_number + 1
                                                    results_dict_saved=False
                                                    logger_saved=False
                                                    results_current_saved=False
                                                    details_saved=False
                                                    path_to_results=mother_output_dir+'run_'+str(run_number)+'/'
                                                    to_run=True
                                                    try:    
                                                        os.mkdir(path_to_results)
                                                    except:
                                                        print(path_to_results+' file exists')
                                                        to_run=False
                                                    try:
                                                        if to_run:
                                                            path_to_weights=path_to_results+'weights/'
                                                            try:
                                                                os.mkdir(path_to_weights)
                                                            except:
                                                                print(path_to_weights+' file exists')
                                                            print("===========\nGenerating train data for run",run_number,".\n")
                                                            print("===========\n")
                                                            start=timer()
                                                            X_data_train=X_data_train_prep_all[:nsamples,:]
                                                            if corr == "corr":
                                                                V = None
                                                            elif corr == "uncorr":
                                                                V = MixtureDistributions.rot_matrix(X_data_train)
                                                                X_data_train = MixtureDistributions.transform_data(X_data_train,V)
                                                                X_data_test = MixtureDistributions.transform_data(X_data_test,V)
                                                            else:
                                                                V = None
                                                            end=timer()
                                                            train_data_time=end-start
                                                            print("Train data generated in",train_data_time,"s.\n")       
                                                            hllabel='-'.join(str(e) for e in hidden_layers)
                                                            Utils.save_hyperparams(path_to_results,hyperparams_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,seed_dist,seed_test)
                                                            print("===========\nRunning",run_number,"/",n_runs,"with hyperparameters:\n",
                                                                  "ndims=",ndims,"\n",
                                                                  "seed=",seed,"\n",
                                                                  "nsamples=",nsamples,"\n",
                                                                  "correlation=",corr,"\n",
                                                                  "activation=",activation,"\n",
                                                                  "eps_regulariser=",eps_regulariser,"\n",
                                                                  "regulariser=",regulariser,"\n",
                                                                  "bijector=",bijector_name,"\n",
                                                                  "nbijectors=",nbijectors,"\n",
                                                                  "spline_knots=",spline_knots,"\n",
                                                                  "range_min=",range_min,"\n",
                                                                  "batch_size=",batch_size,"\n",
                                                                  "hidden_layers=",hidden_layers,
                                                                  "epocs_input=",epochs,
                                                                  "\n===========\n")
                                                            bijector=Bijectors.ChooseBijector(bijector_name,ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style=perm_style)
                                                            Utils.save_bijector_info(bijector,path_to_results)
                                                            base_dist=Distributions.gaussians(ndims)
                                                            nf_dist=tfd.TransformedDistribution(base_dist,bijector)
                                                            start=timer()
                                                            print("Training model.\n")
                                                            epochs_input = epochs
                                                            lr=lr_orig
                                                            n_displays=1
                                                            print("Train first sample:",X_data_train[0])
                                                            history=Trainer.graph_execution(ndims,nf_dist, X_data_train,epochs, batch_size, n_displays,path_to_results,load_weights=True,load_weights_path=path_to_weights,lr=lr,patience=patience,min_delta_patience=min_delta_patience,reduce_lr_factor=lr_change,seed=seed)
                                                            t_losses_all=list(history.history['loss'])
                                                            v_losses_all=list(history.history['val_loss'])
                                                            end=timer()
                                                            epochs_output = len(t_losses_all)
                                                            training_time=end-start
                                                            print("Model trained in",training_time,"s.\n")
                                                            #continue
                                                            start=timer()
                                                            print("===========\nComputing predictions\n===========\n")
                                                            with tf.device('/device:CPU:0'):
                                                                if V is not None:
                                                                    X_data_train = MixtureDistributions.inverse_transform_data(X_data_train,V)
                                                                    X_data_test = MixtureDistributions.inverse_transform_data(X_data_test,V)
                                                                #reload_best
                                                                nf_dist,_=Utils.load_model(nf_dist,path_to_results,ndims,lr=.000001)
                                                                nf_dist=SoftClipTransform(nf_dist,hinge)
                                                                logprob_nf=nf_dist.log_prob(X_data_test).numpy()
                                                                pickle_logprob_nf=open(path_to_results+'logprob_nf.pcl', 'wb')
                                                                pickle.dump(logprob_nf, pickle_logprob_nf, protocol=4)
                                                                pickle_logprob_nf.close()
                                                                X_data_nf=Utils.nf_sample_save(nf_dist,path_to_results,sample_size=ntest_samples,rot=V,iter_size=10000,seed=seed)
                                                                X_data_nf=postprocess_data(X_data_nf,preprocess_params)
                                                                print("Test first sample:",X_data_test[0])
                                                                print("NF first sample:",X_data_nf[0])
                                                                #kl_divergence,ks_median,ks_mean,ad_mean,ad_median,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr=Metrics.ComputeMetrics(X_data_test,X_data_nf)
                                                                dist_1 = tf.cast(X_data_nf,tf.float64)
                                                                dist_2 = tf.cast(X_data_test,tf.float64)
                                                                
                                                                KSres,SWDres=Metrics.GetMetrics(dist_1,dist_2)
                                                                Metrics.SaveNewMetrics(SWDres,'SWD',path_to_results)
                                                                Metrics.SaveNewMetrics(KSres,'KS',path_to_results)
                                                                
                                                                KS_pv_mean,KS_pv_std=Metrics.MeansAndStdMetrics(KSres.get('pvalue_means'))
                                                                KS_st_mean,KS_st_std=Metrics.MeansAndStdMetrics(KSres.get('statistic_means'))
                                                                
                                                                SWDmean,SWDstd=Metrics.MeansAndStdMetrics(SWDres.get('metric_means'))
                                                                
                                                                results_dict=Utils.ResultsToDict(results_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time,KS_pv_mean,KS_pv_std,KS_st_mean,KS_st_std,SWDmean,SWDstd)
                                                                #results_dict=Utils.ResultsToDict(results_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_mean,ad_median,w_distance_median,w_distance_mean,frob_norm,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time)
                                                                results_dict_saved=True
                                                                print("Results dict saved")
                                                                Utils.logger(log_file_name,results_dict)
                                                                logger_saved=True
                                                                print("Logger saved")
                                                                Utils.results_current(path_to_results,results_dict)
                                                                results_current_saved=True
                                                                print("Results saved")
                                                                Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,path_to_results)
                                                                details_saved=True
                                                                print("Details saved")
                                                                Plotters.train_plotter(t_losses_all,v_losses_all,path_to_results)
                                                                corner_start=timer()
                                                                #Plotters.cornerplotter(X_data_test,X_data_nf,path_to_results,ndims,norm=True)
                                                                #Plotters.marginal_plot(X_data_test,X_data_nf,path_to_results,ndims)
                                                                #Plotters.sample_plotter(X_data_test,nf_dist,path_to_results)
                                                                
                                                                
                                                                PlotsandHDPI.GeneratePlotandHDPIResults(X_data_test,X_data_nf,path_to_results,ndims)
                                                                
                                                                end=timer()
                                                                predictions_time=end-start
                                                                print("Model predictions computed in",predictions_time,"s.\n")
                                                        else:
                                                            print("===========\nRun",run_number,"/",n_runs,"already exists. Skipping it.\n")
                                                            print("===========\n")
                                                    except Exception as ex:
                                                        # Get current system exception
                                                        ex_type, ex_value, ex_traceback = sys.exc_info()
                                                        # Extract unformatter stack traces as tuples
                                                        trace_back = traceback.extract_tb(ex_traceback)
                                                        # Format stacktrace
                                                        stack_trace = list()
                                                        for trace in trace_back:
                                                            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                                                        if not results_dict_saved:
                                                            results_dict=Utils.ResultsToDict(results_dict,run_number,seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,hllabel,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time,"nan","nan","nan","nan","nan","nan")
                                                        if not logger_saved:
                                                            Utils.logger(log_file_name,results_dict)
                                                        if not results_current_saved:
                                                            Utils.results_current(path_to_results,results_dict)
                                                        if not details_saved:
                                                            try:
                                                                Utils.save_details_json(hyperparams_dict,results_dict,t_losses_all,v_losses_all,path_to_results)
                                                            except:
                                                                Utils.save_details_json(hyperparams_dict,results_dict,None,None,path_to_results)
                                                                
                                                                
                                                        print("===========\nRun failed\n")
                                                        print("Exception type : %s " % ex_type.__name__)
                                                        print("Exception message : %s" %ex_value)
                                                        print("Stack trace : %s" %stack_trace)
                                                        print("===========\n")   
results_frame=pd.DataFrame(results_dict)
results_frame.to_csv(mother_output_dir+'results_last_run.txt',index=False)
print("Everything done.")
