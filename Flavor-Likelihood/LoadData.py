import numpy as np
from os import path
import pickle


def create_and_save(flavor_dataset_samples_new,flavor_dataset_loglik_values_new,flavor_dataset_logprob_values_new,flavor_dataset_logprior_values_new,X_data_train_file,X_data_test_file,X_data_logprob_train_file,X_data_logprob_test_file,X_data_loglik_test_file,X_data_loglik_train_file,X_data_logprior_test_file,X_data_logprior_train_file):

    nsamples_train=1500000
    nsamples_test=500000
    
    rnd_indices = np.random.choice(np.arange(len(flavor_dataset_samples_new)),size=nsamples_train+nsamples_test,replace=False)


    rnd_indices_train=rnd_indices[:nsamples_train]
    
    rnd_indices_test=rnd_indices[nsamples_train:]

    X_data_train = flavor_dataset_samples_new[rnd_indices_train,:]
    X_data_test = flavor_dataset_samples_new[rnd_indices_test,:]


    X_data_logprob_train = flavor_dataset_logprob_values_new[rnd_indices_train]
    X_data_logprob_test = flavor_dataset_logprob_values_new[rnd_indices_test]
    
    X_data_loglik_train = flavor_dataset_loglik_values_new[rnd_indices_train]
    X_data_loglik_test = flavor_dataset_loglik_values_new[rnd_indices_test]
    
    X_data_logprior_train = flavor_dataset_logprior_values_new[rnd_indices_train]
    X_data_logprior_test = flavor_dataset_logprior_values_new[rnd_indices_test]
    
    
    
    
    print('x data train')
    print(np.shape(X_data_train))
    
    print('x data test')
    print(np.shape(X_data_test))

    
    pickle_train_out = open(X_data_train_file, 'wb')
    pickle.dump(X_data_train, pickle_train_out, protocol=4)
    pickle_train_out.close()

    pickle_test_out = open(X_data_test_file, 'wb')
    pickle.dump(X_data_test, pickle_test_out, protocol=4)
    pickle_train_out.close()
    
    
    
    pickle_logprobs_test = open( X_data_logprob_test_file, 'wb')
    pickle.dump(X_data_logprob_test, pickle_logprobs_test, protocol=4)
    pickle_logprobs_test.close()
    
    pickle_logprobs_train = open( X_data_logprob_train_file, 'wb')
    pickle.dump(X_data_logprob_train, pickle_logprobs_train, protocol=4)
    pickle_logprobs_train.close()
    
    
    pickle_loglik_test = open( X_data_loglik_test_file, 'wb')
    pickle.dump(X_data_loglik_test, pickle_loglik_test, protocol=4)
    pickle_loglik_test.close()
    
    pickle_loglik_train = open( X_data_loglik_train_file, 'wb')
    pickle.dump(X_data_loglik_train, pickle_loglik_train, protocol=4)
    pickle_loglik_train.close()
    
    pickle_logprior_test = open( X_data_logprior_test_file, 'wb')
    pickle.dump(X_data_logprior_test, pickle_logprior_test, protocol=4)
    pickle_loglik_test.close()
    
    pickle_logprior_train = open( X_data_logprior_train_file, 'wb')
    pickle.dump(X_data_logprior_train, pickle_logprior_train, protocol=4)
    pickle_loglik_train.close()
    
    return

def load_data():
    samples_folder_old = "../data/HEPFit_data/Data_old/"
    samples_folder_new = "../data/HEPFit_data/Data_new/"
    if path.exists(samples_folder_old):
        flavor_dataset_old = np.load(path.join(samples_folder_old,"FlavourData.npy"))
    else:
        print("Samples OLD not found")
    if path.exists(samples_folder_new):
        flavor_dataset_new_chains_1 = np.load(path.join(samples_folder_new,"FlavourData_new_1.npy"))#.reshape(30000*240,90)
        flavor_dataset_new_chains_2 = np.load(path.join(samples_folder_new,"FlavourData_new_2.npy"))#.reshape(30000*240,90)
        flavor_dataset_new_chains_3 = np.load(path.join(samples_folder_new,"FlavourData_new_3.npy"))#.reshape(30000*240,90)
    else:
        print("Samples NEW not found")

    print(np.shape(flavor_dataset_old))
    print(np.shape(flavor_dataset_new_chains_1))
    print(np.shape(flavor_dataset_new_chains_2))
    print(np.shape(flavor_dataset_new_chains_3))


    flavor_dataset_new_chains = np.concatenate((flavor_dataset_new_chains_1[10000::6,:,:],flavor_dataset_new_chains_2[10000::6,:,:],flavor_dataset_new_chains_3[10000::6,:,:]))
    np.shape(flavor_dataset_new_chains)


    print(np.shape(flavor_dataset_new_chains))
    flavor_dataset_flat_new = flavor_dataset_new_chains.reshape(10002*240,91)
    flavor_dataset_samples_new = flavor_dataset_flat_new[:2400000,:89]
    flavor_dataset_loglik_values_new = flavor_dataset_flat_new[:2400000,89]
    flavor_dataset_logprob_values_new = flavor_dataset_flat_new[:2400000,90]
    flavor_dataset_logprior_values_new = flavor_dataset_logprob_values_new-flavor_dataset_loglik_values_new
    print("Samples new:",[np.shape(flavor_dataset_samples_new),np.shape(flavor_dataset_logprob_values_new),np.shape(flavor_dataset_loglik_values_new),np.shape(flavor_dataset_logprior_values_new)])
    
    
    return flavor_dataset_samples_new,flavor_dataset_loglik_values_new,flavor_dataset_logprob_values_new,flavor_dataset_logprior_values_new




X_data_train_file='../data/HEPFit_data/Data_new/X_data_train_flavor_new_1p5M'
X_data_test_file='../data/HEPFit_data/Data_new/X_data_test_flavor_new_500K'

X_data_logprob_train_file='../data/HEPFit_data/Data_new/X_data_train_logprob_flavor_new_1p5M'
X_data_logprob_test_file='../data/HEPFit_data/Data_new/X_data_test_logprob_flavor_new_500k'

X_data_loglik_test_file='../data/HEPFit_data/Data_new/X_data_test_loglike_flavor_new_500k'
X_data_loglik_train_file='../data/HEPFit_data/Data_new/X_data_train_loglike_flavor_new_1p5M'

X_data_logprior_test_file='../data/HEPFit_data/Data_new/X_data_test_logprior_flavor_new_500k'
X_data_logprior_train_file='../data/HEPFit_data/Data_new/X_data_train_logprior_flavor_new_1p5M'





flavor_dataset_samples_new,flavor_dataset_loglik_values_new,flavor_dataset_logprob_values_new,flavor_dataset_logprior_values_new=load_data()

create_and_save(flavor_dataset_samples_new,flavor_dataset_loglik_values_new,flavor_dataset_logprob_values_new,flavor_dataset_logprior_values_new,X_data_train_file,X_data_test_file,X_data_logprob_train_file,X_data_logprob_test_file,X_data_loglik_test_file,X_data_loglik_train_file,X_data_logprior_test_file,X_data_logprior_train_file)



