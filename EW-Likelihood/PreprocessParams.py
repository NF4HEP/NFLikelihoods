
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from timeit import default_timer as timer
import matplotlib.lines as mlines

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


def corner_plot(samples,name):

    #samples=tf.transpose(samples)


    blue_line = mlines.Line2D([], [], color='blue', label='target')
    figure=corner.corner(samples,color='blue')
    #plt.legend(handles=[blue_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    plt.savefig(name,pil_kwargs={'quality':50})
    plt.close()
    
    return
    
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
    

def save_preprocess_params(preprocess_params,preporcess_params_path):

    with open(preporcess_params_path,'wb') as file:
        pickle.dump(preprocess_params,file)

    return

def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    return preprocess_params


X_data_train_file = 'data/X_data_EW_2_500k_1'
X_data_test_file = 'data/X_data_test_EW_2_300k_1'
logprobs_data_test_file = 'data/Y_data_test_EW_2_300k_1'

X_data_train,X_data_test,logprobs_data_test=loadData(X_data_train_file,X_data_test_file,logprobs_data_test_file)
X_data_test=X_data_test[:100000,:]
logprobs_data_test=logprobs_data_test[:100000,0]
preporcess_params_path='preprocess_data_ewfit2.pcl'



print('generating and svaing preprocess params')
means=np.mean(X_data_train,axis=0)
print(means)
stds=np.std(X_data_train,axis=0)
print(stds)
preprocess_params={'means':means,'stds':stds}

save_preprocess_params(preprocess_params,preporcess_params_path)


preprocess_params=load_preprocess_params(preporcess_params_path)
print(preprocess_params)
print('preprocessing')
X_data_test_processed=preprocess_data(X_data_test,preprocess_params)
#X_data_test_postprocessed=postprocess_data(X_data_test_processed,preprocess_params)
#print(np.max(X_data_test_postprocessed[:,:2],axis=0))

print('producing plot')
name='corner_plot_ewfit_preprocess.png'
corner_plot(X_data_test_processed,name)
