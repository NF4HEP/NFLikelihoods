
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from timeit import default_timer as timer
import matplotlib.lines as mlines




def GetData(X_data_file_train,X_data_file_test):

    

    if os.path.exists(X_data_file_train):
        #### import target distribution #######
        print("Importing X_data from file",X_data_file_train)
        pickle_train = open(X_data_file_train,'rb')
        
        start = timer()
        statinfo = os.stat(X_data_file_train)
        X_data_train = pickle.load(pickle_train)
        pickle_train.close()
        pickle_test=open(X_data_file_test,'rb')
        X_data_test = pickle.load(pickle_test)
        pickle_test.close()
        end = timer()
   
        print('File loaded in ',end-start,' seconds.\nFile size is ',statinfo.st_size,'.')
        ### generate target distribtution from samples #######
    return X_data_train,X_data_test




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

def maxmins_preprocessed_data(X_data_test_processed):


    print('#######MAX#######')
    print(list(np.max(X_data_test_processed,axis=0)))
    print('#######MIN#######')
    print(list(np.min(X_data_test_processed,axis=0)))

    return

def max_mins():



    max_conditions=[5.6119569040384665, 9.408321669056832, 5.921143923401007, 2.0987539948122715, 1.7335918107131771, 1.5496947863239248, 6.015819026904569, 5.357000788249108, 5.078956527236209, 1.9706613161898978, 1.9101265446366096, 1.6733549612022363, 5.136582353186947, 5.450362313377298, 2.0646370061806674, 1.9235362337821624, 1.7261000227703496, 1.7584699885882422, 1.7261145162920541, 1.734678341121733, 5.303146000538385, 1.2408237363374655, 4.5208285276014655, 4.611096424211525, 4.67247424059945, 1.4438519838181176, 1.7425589680115732, 1.6612583351965835, 1.7815774951237893, 1.738228427666701, 4.750417146123906, 4.772144294967443, 4.098516986699341, 4.687790767222508, 4.648232436632133, 4.73335050833763, 4.992481060191991, 4.5685777884876435, 4.692557719410474, 4.6848372177608635, 5.217863521870644, 4.652724547849794, 4.613649584431962, 4.648134663110919, 4.982229325918564, 4.422509818819, 4.439536968846636, 4.807716213331392, 4.914197213991386, 5.085327218848985, 5.209548924005082, 4.733598946040475, 4.439572029870957, 5.159573608150739, 4.434700977430681, 4.840175450673164, 5.392032812553973, 5.141916761568047, 5.011726785228644, 4.670226661815609, 4.955455036694929, 5.042015303366961, 4.412874228275154, 4.710184677756211, 4.912336401667579, 4.819064018516058, 4.416760141828074, 5.172366091120923, 4.670204816478531, 5.017447935350553, 4.888395493928422, 4.587133254342274, 4.964054682574907, 4.766753149198907, 4.384927585158509, 4.915176688322467, 4.520501373940944, 2.9103100032749665, 5.586215137792808, 3.2856681007051116, 3.4139782803045717, 2.147317265602916, 2.260629889692233, 3.477996125332174, 4.904049555630162, 2.7418604159447146, 3.9871097614145503, 2.123393534531489, 2.1199681188827193]
    
    min_conditions=[-1.5084377926591506, -1.1403203148728418, -2.496500952464059, -3.4063185703305665, -1.8244945529911047, -1.7465928259936245, -1.1664721288961046, -1.4121997300790023, -1.3784733701350902, -2.043872394062949, -2.415854255775898, -1.975487952321519, -1.6098315102808707, -1.3127420122099642, -3.2462309463922785, -2.1201205998294044, -1.7534069948736217, -1.7530192327414973, -1.7105445832971058, -1.7308037311615643, -1.1121057269957326, -2.1710137373887326, -4.6875861871041, -4.2721357126903365, -4.536547590491557, -2.6782964764941752, -1.7161104528406887, -1.7718911017041539, -1.6826392859341341, -1.7312847793305364, -4.335857116055725, -4.439620071795575, -3.66228858986652, -4.770938296719519, -4.54863749293239, -4.993337476228878, -4.835147802468657, -4.782566882850699, -4.600584766861787, -4.708251144639543, -4.5494115434656806, -4.686817053828232, -4.704545726595088, -5.190751195092813, -4.520832661523522, -4.537798035713625, -4.284177009139489, -4.681677266614382, -4.565935528255109, -4.7353084066141795, -4.706371837702957, -4.556178308190562, -4.78790250770006, -4.843064350859609, -4.669400658073029, -4.638740107100772, -4.398275377744491, -4.8102190132314835, -4.949473271379748, -4.669071920330874, -5.485755070769344, -4.693172065282441, -4.460176876493423, -4.848101524340766, -4.279124191441311, -4.828178614796366, -4.420692411259104, -4.8010547556168435, -4.325342737868171, -4.816959489897645, -5.072689643860321, -4.170787843285713, -4.342952438293287, -4.762433021903761, -4.376852083237983, -4.531748269282737, -4.998387076297546, -1.9253108978562523, -4.358093562823174, -4.87016585330077, -3.5672813718258976, -2.1257475777616324, -1.9168860977284283, -4.240789174065665, -4.7142165879651134, -1.990144028197836, -4.239242098184308, -2.129705565636902, -1.9856780689375653]



    return


X_data_file_train = '../data/HEPFit_data/Data_new/X_data_train_flavor_new_1p5M'
X_data_file_test = '../data/HEPFit_data/Data_new/X_data_test_flavor_new_500K'
preporcess_params_path='preprocess_data_flavorfit_new.pcl'

X_data_train,X_data_test=GetData(X_data_file_train,X_data_file_test)


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
maxmins_preprocessed_data(X_data_test_processed)


print('producing plot')
name='corner_plot_flavorfit_preprocess_2.png'
corner_plot(X_data_test_processed,name)
