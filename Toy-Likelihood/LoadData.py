import pickle
import numpy as np
import os




def create_data_files(input_samples_file,X_data_train_file,X_data_test_file,logprobs_test_file,nsamples_train,nsamples_test):

    print("Importing samples from file",input_samples_file)
    pickle_in = open(input_samples_file,'rb')
    allsamples = pickle.load(pickle_in)
    logprob_values = pickle.load(pickle_in)
 
    #EWfit_dataset = EWfit_dataset.reshape(shape[0]*shape[1],shape[2])        print(np.shape(EWfit_dataset))
    rnd_indices_train = np.random.choice(np.arange(len(allsamples)),size=nsamples_train,replace=False)
    rnd_indices_test = np.random.choice(np.arange(len(allsamples)),size=nsamples_test,replace=False)
    X_data_train = allsamples[rnd_indices_train,:]
    
    X_data_test = allsamples[rnd_indices_test,:]
    logprobs_data_test=logprob_values[rnd_indices_test]
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


#The pickle file most be downloaded from
input_samples_file = 'data/likelihood_unbiased_sm_13_thinned1000_11M.pickle'


##Path to train and test data
X_data_train_file = 'data/X_data_LF100_2M'
X_data_test_file = 'data/X_data_test_LF100_500000'
logprobs_test_file = 'data/logprobs_data_test_LF100_500000'


###number of total training and test samples.
nsamples_train=2000000
nsamples_test=500000

create_data_files(input_samples_file,X_data_train_file,X_data_test_file,logprobs_test_file,nsamples_train,nsamples_test)


