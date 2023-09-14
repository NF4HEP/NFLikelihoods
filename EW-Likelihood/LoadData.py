import pickle
import numpy as np
import os




def create_data_files(input_samples_file,X_data_train_file,X_data_test_file,logprobs_test_file,nsamples_train,nsamples_test):

    print("Importing samples from file",input_samples_file)
    EWfit_dataset = np.load(input_samples_file)
    shape=np.shape(EWfit_dataset)
    print(shape)

    EWfit_dataset = EWfit_dataset.reshape(shape[0]*shape[1],shape[2])
    EWfit_dataset=EWfit_dataset[(EWfit_dataset[:,10] >0) & (EWfit_dataset[:,11] > 0)]
    rnd_indices_train = np.random.choice(np.arange(len(EWfit_dataset)),size=nsamples_train,replace=False)
    rnd_indices_test = np.random.choice(np.arange(len(EWfit_dataset)),size=nsamples_test,replace=False)
    X_data_train = EWfit_dataset[rnd_indices_train,:-2]
    data_test = EWfit_dataset[rnd_indices_test,:]
    logprobs_data_test = data_test[:,-2:]
    X_data_test = data_test[:,:-2]
    
    
    
    X_data_train = np.delete(X_data_train, 28, 1)
    X_data_test= np.delete(X_data_test, 28, 1)
    
    
    print('x data train')
    print(np.shape(X_data_train))
    
    print('x data test')
    print(np.shape(X_data_test))
    print('logprobs test')
    print(np.shape(logprobs_data_test))
    print(logprobs_data_test[:10])
    
    pickle_train_out = open(X_data_train_file, 'wb')
    pickle.dump(X_data_train, pickle_train_out, protocol=4)
    pickle_train_out.close()

    pickle_test_out = open(X_data_test_file, 'wb')
    pickle.dump(X_data_test, pickle_test_out, protocol=4)
    pickle_train_out.close()
    
    pickle_logprobs_test = open(logprobs_test_file, 'wb')
    pickle.dump(logprobs_data_test, pickle_logprobs_test, protocol=4)
    pickle_logprobs_test.close()

    return


##The input samples file can be downloaded from...
input_samples_file = '../data/data_ew_new/EWData_New_Mw.npy'
X_data_train_file = '../data/X_data_EW_2_500k_1'
X_data_test_file = '../data/X_data_test_EW_2_300k_1'
logprobs_test_file = '../data/Y_data_test_EW_2_300k_1'

##number of samples
nsamples_train=500000
nsamples_test=300000

create_data_files(input_samples_file,X_data_train_file,X_data_test_file,logprobs_test_file,nsamples_train,nsamples_test)


