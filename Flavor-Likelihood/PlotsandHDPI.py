import arviz as az
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats
from statistics import mean,median
import pickle
import os
from timeit import default_timer as timer
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import corner





def load_preprocess_params(preporcess_params_path):
    with open(preporcess_params_path,'rb') as file:
        preprocess_params=pickle.load(file)
    
    return preprocess_params


def postprocess_data(data,preprocess_params):

    means=preprocess_params.get('means')
    stds=preprocess_params.get('stds')
    
    postprocess_data=data*stds+means
    
    return postprocess_data


def CutMaxMin(nf_dist):
    column_names=np.arange(89).tolist()
    for j in range(len(column_names)):
        column_names[j]=str(column_names[j])
    print(column_names)
    


    df = pd.DataFrame(data=nf_dist,columns=column_names)

    max_conditions=[5.6119569040384665, 9.408321669056832, 5.921143923401007, 2.0987539948122715, 1.7335918107131771, 1.5496947863239248, 6.015819026904569, 5.357000788249108, 5.078956527236209, 1.9706613161898978, 1.9101265446366096, 1.6733549612022363, 5.136582353186947, 5.450362313377298, 2.0646370061806674, 1.9235362337821624, 1.7261000227703496, 1.7584699885882422, 1.7261145162920541, 1.734678341121733, 5.303146000538385, 1.2408237363374655, 4.5208285276014655, 4.611096424211525, 4.67247424059945, 1.4438519838181176, 1.7425589680115732, 1.6612583351965835, 1.7815774951237893, 1.738228427666701, 4.750417146123906, 4.772144294967443, 4.098516986699341, 4.687790767222508, 4.648232436632133, 4.73335050833763, 4.992481060191991, 4.5685777884876435, 4.692557719410474, 4.6848372177608635, 5.217863521870644, 4.652724547849794, 4.613649584431962, 4.648134663110919, 4.982229325918564, 4.422509818819, 4.439536968846636, 4.807716213331392, 4.914197213991386, 5.085327218848985, 5.209548924005082, 4.733598946040475, 4.439572029870957, 5.159573608150739, 4.434700977430681, 4.840175450673164, 5.392032812553973, 5.141916761568047, 5.011726785228644, 4.670226661815609, 4.955455036694929, 5.042015303366961, 4.412874228275154, 4.710184677756211, 4.912336401667579, 4.819064018516058, 4.416760141828074, 5.172366091120923, 4.670204816478531, 5.017447935350553, 4.888395493928422, 4.587133254342274, 4.964054682574907, 4.766753149198907, 4.384927585158509, 4.915176688322467, 4.520501373940944, 2.9103100032749665, 5.586215137792808, 3.2856681007051116, 3.4139782803045717, 2.147317265602916, 2.260629889692233, 3.477996125332174, 4.904049555630162, 2.7418604159447146, 3.9871097614145503, 2.123393534531489, 2.1199681188827193]
        
    min_conditions=[-1.5084377926591506, -1.1403203148728418, -2.496500952464059, -3.4063185703305665, -1.8244945529911047, -1.7465928259936245, -1.1664721288961046, -1.4121997300790023, -1.3784733701350902, -2.043872394062949, -2.415854255775898, -1.975487952321519, -1.6098315102808707, -1.3127420122099642, -3.2462309463922785, -2.1201205998294044, -1.7534069948736217, -1.7530192327414973, -1.7105445832971058, -1.7308037311615643, -1.1121057269957326, -2.1710137373887326, -4.6875861871041, -4.2721357126903365, -4.536547590491557, -2.6782964764941752, -1.7161104528406887, -1.7718911017041539, -1.6826392859341341, -1.7312847793305364, -4.335857116055725, -4.439620071795575, -3.66228858986652, -4.770938296719519, -4.54863749293239, -4.993337476228878, -4.835147802468657, -4.782566882850699, -4.600584766861787, -4.708251144639543, -4.5494115434656806, -4.686817053828232, -4.704545726595088, -5.190751195092813, -4.520832661523522, -4.537798035713625, -4.284177009139489, -4.681677266614382, -4.565935528255109, -4.7353084066141795, -4.706371837702957, -4.556178308190562, -4.78790250770006, -4.843064350859609, -4.669400658073029, -4.638740107100772, -4.398275377744491, -4.8102190132314835, -4.949473271379748, -4.669071920330874, -5.485755070769344, -4.693172065282441, -4.460176876493423, -4.848101524340766, -4.279124191441311, -4.828178614796366, -4.420692411259104, -4.8010547556168435, -4.325342737868171, -4.816959489897645, -5.072689643860321, -4.170787843285713, -4.342952438293287, -4.762433021903761, -4.376852083237983, -4.531748269282737, -4.998387076297546, -1.9253108978562523, -4.358093562823174, -4.87016585330077, -3.5672813718258976, -2.1257475777616324, -1.9168860977284283, -4.240789174065665, -4.7142165879651134, -1.990144028197836, -4.239242098184308, -2.129705565636902, -1.9856780689375653]





    for j in range(len(max_conditions)):

        df=df.loc[(df[column_names[j]]<max_conditions[j]) & (df[column_names[j]]>min_conditions[j])]

    selected_data=df.to_numpy()
    return selected_data

def ImportNFdata(path_to_nf_sample):

        ###pickle test
        print("Importing nf_sample from file",path_to_nf_sample)
        
        
        #pickle_test = open(path_to_nf_sample,'rb')
        #start = timer()
        #X_data_test = pickle.load(pickle_test)
        #print(np.shape(X_data_test))
        #pickle_test.close()
        X_data_test=np.load(path_to_nf_sample,allow_pickle=False)

        return X_data_test
    





def ImportTrueData(X_data_test_file):

        ###pickle test
        print("Importing X_data_test from file",X_data_test_file)
        pickle_test = open(X_data_test_file,'rb')
        start = timer()
        X_data_test = pickle.load(pickle_test)
        print(np.shape(X_data_test))
        pickle_test.close()


        return X_data_test


def HDPI(posterior_samples):


    hdpi1 = az.hdi(posterior_samples, hdi_prob=0.68,multimodal=True)
    hdpi2 = az.hdi(posterior_samples, hdi_prob=0.954,multimodal=True)
    hdpi3 = az.hdi(posterior_samples, hdi_prob=0.997,multimodal=True)

    return list(hdpi1),list(hdpi2),list(hdpi3)


def HDPIOverDims(full_dim_samples):

    ndims=np.shape(full_dim_samples)[1]
    
    hdpis_dict={'hdpi_1sigma':[],'hdpi_2sigma':[],'hdpi_3sigma':[]}
    
    
    
    for k in range(ndims):
        hdpi1,hdpi2,hdpi3=HDPI(full_dim_samples[:,k])
        
        hdpis_dict.get('hdpi_1sigma').append(hdpi1)
        hdpis_dict.get('hdpi_2sigma').append(hdpi2)
        hdpis_dict.get('hdpi_3sigma').append(hdpi3)
    
    print(hdpis_dict)


    return hdpis_dict




def ComputeHDPIDif(hdpis_dict1,hdpis_dict2):

    hdpis_dict_difs={'hdpi_1sigma':[],'hdpi_2sigma':[],'hdpi_3sigma':[]}
    
    
    ndims=len(hdpis_dict1.get('hdpi_1sigma'))
    
    
    hdpi_data1_sigma1=hdpis_dict1.get('hdpi_1sigma')
    hdpi_data2_sigma1=hdpis_dict2.get('hdpi_1sigma')
    hdpi_data1_sigma2=hdpis_dict1.get('hdpi_2sigma')
    hdpi_data2_sigma2=hdpis_dict2.get('hdpi_2sigma')
    hdpi_data1_sigma3=hdpis_dict1.get('hdpi_3sigma')
    hdpi_data2_sigma3=hdpis_dict2.get('hdpi_3sigma')
    
    
    
    for k in range(ndims):

        
        hdpi_dif1=[a_i - b_i for a_i, b_i in zip(hdpi_data1_sigma1[k], hdpi_data2_sigma1[k])]
        hdpi_dif1 =  [abs(_) for _ in hdpi_dif1]
 
        hdpis_dict_difs.get('hdpi_1sigma').append(hdpi_dif1)
        print(hdpi_dif1)
        
        
        if k==0:
            print('abs difs')
            print(hdpi_dif1)
      
        #print(hdpi_dif2)
        hdpi_dif2=[a_i - b_i for a_i, b_i in zip(hdpi_data1_sigma2[k], hdpi_data2_sigma2[k])]
        
        hdpi_dif2 =  [abs(_) for _ in hdpi_dif2]
        hdpis_dict_difs.get('hdpi_2sigma').append(hdpi_dif2)
        
        
        hdpi_dif3=[a_i - b_i for a_i, b_i in zip(hdpi_data1_sigma3[k], hdpi_data2_sigma3[k])]
        hdpi_dif3 =  [abs(_) for _ in hdpi_dif3]
        
        hdpis_dict_difs.get('hdpi_3sigma').append(hdpi_dif3)
        
    return hdpis_dict_difs
    

def ComputeHDPIRelDif(hdpis_dict1,hdpis_dict_difs):

    hdpis_dict_rel_difs={'hdpi_1sigma':[],'hdpi_2sigma':[],'hdpi_3sigma':[],'mean_hdpi_1sigma':[],'mean_hdpi_2sigma':[],'mean_hdpi_3sigma':[]}
    
    
    ndims=len(hdpis_dict1.get('hdpi_1sigma'))
    
    
    hdpi_data1_sigma1=hdpis_dict1.get('hdpi_1sigma')
    hdpi_difs_sigma1=hdpis_dict_difs.get('hdpi_1sigma')
    hdpi_data1_sigma2=hdpis_dict1.get('hdpi_2sigma')
    hdpi_difs_sigma2=hdpis_dict_difs.get('hdpi_2sigma')
    hdpi_data1_sigma3=hdpis_dict1.get('hdpi_3sigma')
    hdpi_difs_sigma3=hdpis_dict_difs.get('hdpi_3sigma')



    for k in range(ndims):
        
        hdpi_rel_dif1=[abs(b_i/a_i) for a_i, b_i in zip(hdpi_data1_sigma1[k], hdpi_difs_sigma1[k])]
        hdpis_dict_rel_difs.get('hdpi_1sigma').append(hdpi_rel_dif1)
        print('before means')
        print(hdpi_rel_dif1)
        hdpi_rel_dif1=[np.mean(a_i) for a_i in hdpi_rel_dif1]
        print(hdpi_rel_dif1)
        hdpis_dict_rel_difs.get('mean_hdpi_1sigma').append(mean(hdpi_rel_dif1))
        
        
        if k==0:
            print('rel difs')
           
            
        
        
        hdpi_rel_dif2=[abs(b_i/a_i) for a_i, b_i in zip(hdpi_data1_sigma2[k], hdpi_difs_sigma2[k])]
        hdpis_dict_rel_difs.get('hdpi_2sigma').append(hdpi_rel_dif2)
        hdpi_rel_dif2=[np.mean(a_i) for a_i in hdpi_rel_dif2]
        hdpis_dict_rel_difs.get('mean_hdpi_2sigma').append(mean(hdpi_rel_dif2))
        
        hdpi_rel_dif3=[abs(b_i/a_i) for a_i, b_i in zip(hdpi_data1_sigma3[k], hdpi_difs_sigma3[k])]
        hdpis_dict_rel_difs.get('hdpi_3sigma').append(hdpi_rel_dif3)
        hdpi_rel_dif3=[np.mean(a_i) for a_i in hdpi_rel_dif3]
        hdpis_dict_rel_difs.get('mean_hdpi_3sigma').append(mean(hdpi_rel_dif3))


    return hdpis_dict_rel_difs

def GetMeansAndMedians(hdpis_dict_rel_difs):

    means_sigma1=hdpis_dict_rel_difs.get('mean_hdpi_1sigma')
    big_mean_sigma1=mean(means_sigma1)
    big_median_sigma1=median(means_sigma1)
    
    means_sigma2=hdpis_dict_rel_difs.get('mean_hdpi_2sigma')
    big_mean_sigma2=mean(means_sigma2)
    big_median_sigma2=median(means_sigma2)
    
    means_sigma3=hdpis_dict_rel_difs.get('mean_hdpi_3sigma')
    big_mean_sigma3=mean(means_sigma3)
    big_median_sigma3=median(means_sigma3)
    
    print('sigma1')
    print(big_mean_sigma1)
    print(big_median_sigma1)
    
    print('sigma2')
    print(big_mean_sigma2)
    print(big_median_sigma2)
    
    print('sigma3')
    print(big_mean_sigma3)
    print(big_median_sigma3)
    
    
    return big_mean_sigma1,big_median_sigma1,big_mean_sigma2,big_median_sigma2,big_mean_sigma3,big_median_sigma3
    


def DrawCIs(figure,hdpis_dict_true,hdpis_dict_nf,ndims,selection_list,labels_3):


    
    ##Get sigmas lists
    hdpi_data_true_sigma1=hdpis_dict_true.get('hdpi_1sigma')
    hdpi_data_nf_sigma1=hdpis_dict_nf.get('hdpi_1sigma')
    hdpi_data_true_sigma2=hdpis_dict_true.get('hdpi_2sigma')
    hdpi_data_nf_sigma2=hdpis_dict_nf.get('hdpi_2sigma')
    hdpi_data_true_sigma3=hdpis_dict_true.get('hdpi_3sigma')
    hdpi_data_nf_sigma3=hdpis_dict_nf.get('hdpi_3sigma')

    #   Extract the axes
    axes = np.array(figure.axes).reshape((ndims, ndims))

  
    hdpi_data_true_sigma1_sel=list( hdpi_data_true_sigma1[j] for j in selection_list )
    hdpi_data_true_sigma2_sel=list( hdpi_data_true_sigma2[j] for j in selection_list )
    hdpi_data_true_sigma3_sel=list( hdpi_data_true_sigma3[j] for j in selection_list )
    hdpi_data_nf_sigma1_sel=list( hdpi_data_nf_sigma1[j] for j in selection_list )
    hdpi_data_nf_sigma2_sel=list( hdpi_data_nf_sigma2[j] for j in selection_list )
    hdpi_data_nf_sigma3_sel=list( hdpi_data_nf_sigma3[j] for j in selection_list )

    # Loop over the diagonal
    for i in range(ndims):
        ax = axes[i, i]
        ax.set_title(labels_3[i],loc='left')
        print(hdpi_data_true_sigma1[i])
        for j in range(len(hdpi_data_true_sigma1_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma1_sel[i][j][0], color="r",ls='-')
            ax.axvline(hdpi_data_true_sigma1_sel[i][j][1], color="r",ls='-')
        
        for j in range(len(hdpi_data_nf_sigma1_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma1_sel[i][j][0], color="b",ls='-')
            ax.axvline(hdpi_data_nf_sigma1_sel[i][j][1], color="b",ls='-')
            
        for j in range(len(hdpi_data_true_sigma2_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma2_sel[i][j][0], color="r",ls='--')
            ax.axvline(hdpi_data_true_sigma2_sel[i][j][1], color="r",ls='--')
        
        for j in range(len(hdpi_data_nf_sigma2_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma2_sel[i][j][0], color="b",ls='--')
            ax.axvline(hdpi_data_nf_sigma2_sel[i][j][1], color="b",ls='--')
        
        for j in range(len(hdpi_data_true_sigma3_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma3_sel[i][j][0], color="r",ls='-.')
            ax.axvline(hdpi_data_true_sigma3_sel[i][j][1], color="r",ls='-.')
        
        for j in range(len(hdpi_data_nf_sigma3_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma3_sel[i][j][0], color="b",ls='-.')
            ax.axvline(hdpi_data_nf_sigma3_sel[i][j][1], color="b",ls='-.')
            
            
            
            
        '''
        ax.axvline(hdpi_data_true_sigma1[i][0], color="r",ls='-')
        ax.axvline(hdpi_data_true_sigma1[15+i][1], color="r",ls='-')
        
        
        ax.axvline(hdpi_data_nf_sigma1[15+i][0], color="b",ls='-')
        ax.axvline(hdpi_data_nf_sigma1[15+i][1], color="b",ls='-')
        
        
        ax.axvline(hdpi_data_true_sigma2[15+i][0], color="r",ls='--')
        ax.axvline(hdpi_data_true_sigma2[15+i][1], color="r",ls='--')
        ax.axvline(hdpi_data_nf_sigma2[15+i][0], color="b",ls='--')
        ax.axvline(hdpi_data_nf_sigma2[15+i][1], color="b",ls='--')
        
        
        ax.axvline(hdpi_data_true_sigma3[15+i][0], color="r",ls='-.')
        ax.axvline(hdpi_data_true_sigma3[15+i][1], color="r",ls='-.')
        ax.axvline(hdpi_data_nf_sigma3[15+i][0], color="b",ls='-.')
        ax.axvline(hdpi_data_nf_sigma3[15+i][1], color="b",ls='-.')
        '''


    return figure

'''
def marginal_plot(target_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_to_plot,ndims):

    labels = [r'$|h_{0}^{(0)}|$',r'$|h_{+}^{(0)}|$',r'$|h_{-}^{(0)}|$',r'$\\arg\\left(h_{0}^{(0)}\\right)$',r'$\\arg\\left(h_{+}^{(0)}\\right)$',r'$\\arg\\left(h_{-}^{(0)}\\right)$',
 r'$|h_{0}^{(1)}|$',r'$|h_{+}^{(1)}|$',r'$|h_{-}^{(1)}|$',r'$\\arg\\left(h_{0}^{(1)}\\right)$',r'$\\arg\\left(h_{+}^{(1)}\\right)$',r'$\\arg\\left(h_{-}^{(1)}\\right)$',
 r'$|h_{+}^{(1)}|$',r'$|h_{-}^{(1)}|$',r'$\\arg\\left(h_{+}^{(1)}\\right)$',r'$\\arg\\left(h_{-}^{(1)}\\right)$',
 r'$|h_{0}^{(\\text{MP})}|$',r'$\\arg\\left(h_{0}^{(\\text{MP})}\\right)$',r'$|h_{1}^{(\\text{MP})}|$',r'$\\arg\\left(h_{1}^{(\\text{MP})}\\right)$',r'$|h_{2}^{(\\text{MP})}|$',r'$\\arg\\left(h_{1}^{(\\text{MP})}\\right)$',
 r'$F_{Bs}/F_{Bd}$',r'$\\delta Gs_{Gs}$',r'$F_{Bs}$',r'$\\lambda_{8}$',r'$\\alpha_{1}(K^{*})$',r'$\\alpha_{2}(K^{*})$',r'$\\alpha_{2}(\\phi)$',r"$\\alpha_{1}(K^{'})$",
 r'$LCSRFF_{1}$',r'$LCSRFF_{2}$',r'$LCSRFF_{3}$',r'$LCSRFF_{4}$',r'$LCSRFF_{5}$',r'$LCSRFF_{6}$',r'$LCSRFF_{7}$',r'$LCSRFF_{8}$',r'$LCSRFF_{9}$',r'$LCSRFF_{10}$',
 r'$LCSRFF_{11}$',r'$LCSRFF_{12}$',r'$LCSRFF_{13}$',r'$LCSRFF_{14}$',r'$LCSRFF_{15}$',r'$LCSRFF_{16}$',r'$LCSRFF_{17}$',r'$LCSRFF_{18}$',r'$LCSRFF_{19}$',
 r'$LCSRFF_{\\phi 1}$',r'$LCSRFF_{\\phi 2}$',r'$LCSRFF_{\\phi 3}$',r'$LCSRFF_{\\phi 4}$',r'$LCSRFF_{\\phi 5}$',r'$LCSRFF_{\\phi 6}$',r'$LCSRFF_{\\phi 7}$',r'$LCSRFF_{\\phi 8}$',r'$LCSRFF_{\\phi 9}$',r'$LCSRFF_{\\phi 10}$',
 r'$LCSRFF_{\\phi 11}$',r'$LCSRFF_{\\phi 12}$',r'$LCSRFF_{\\phi 13}$',r'$LCSRFF_{\\phi 14}$',r'$LCSRFF_{\\phi 15}$',r'$LCSRFF_{\\phi 16}$',r'$LCSRFF_{\\phi 17}$',r'$LCSRFF_{\\phi 18}$',r'$LCSRFF_{\\phi 19}$',
 r'$LATFF_{11}$',r'$LATFF_{12}$',r'$LATFF_{13}$',r'$LATFF_{14}$',r'$LATFF_{15}$',r'$LATFF_{16}$',r'$LATFF_{17}$',r'$LATFF_{18}$',r'$LATFF_{19}$',
 r'$c^{LQ}_{2223}$',r'$c^{ed}_{2223}$',r'$c^{Ld}_{2223}$',r'$c^{Qe}_{2223}$',r'$c^{LeqQ}_{2223}$',r'$c^{\\prime\\,LeqQ}_{2223}$']


    print('ehllo marginal')
    print(ndims)
    
    
        ##Get sigmas lists
    hdpi_data_true_sigma1=hdpis_dict_true.get('hdpi_1sigma')
    hdpi_data_nf_sigma1=hdpis_dict_nf.get('hdpi_1sigma')
    hdpi_data_true_sigma2=hdpis_dict_true.get('hdpi_2sigma')
    hdpi_data_nf_sigma2=hdpis_dict_nf.get('hdpi_2sigma')
    hdpi_data_true_sigma3=hdpis_dict_true.get('hdpi_3sigma')
    hdpi_data_nf_sigma3=hdpis_dict_nf.get('hdpi_3sigma')
    
    
    
    n_bins=50
    fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=False)


      
    
    for dim in range(ndims):
    
        row=int(dim/4)
        column=int(dim%4)
        
        
        print(dim)
        axs[row,column].hist(target_samples[:,dim], bins=n_bins,density=True,histtype='step',color='red')
        axs[row,column].hist(nf_samples[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
      
        
        axs[row,column].text(.1,1.04,labels[dim],
        horizontalalignment='center',
        transform=axs[row,column].transAxes,fontsize=5)
        
        x_axis = axs[row,column].axes.get_xaxis()
        x_axis.set_visible(False)
        y_axis = axs[row,column].axes.get_yaxis()
        y_axis.set_visible(False)
        
        
        
        axs[row,column].axvline(hdpi_data_true_sigma1[dim][0], color="r",ls='-')
        axs[row,column].axvline(hdpi_data_true_sigma1[dim][1], color="r",ls='-')
        axs[row,column].axvline(hdpi_data_nf_sigma1[dim][0], color="b",ls='-')
        axs[row,column].axvline(hdpi_data_nf_sigma1[dim][1], color="b",ls='-')
        
        
        axs[row,column].axvline(hdpi_data_true_sigma2[dim][0], color="r",ls='--')
        axs[row,column].axvline(hdpi_data_true_sigma2[dim][1], color="r",ls='--')
        axs[row,column].axvline(hdpi_data_nf_sigma2[dim][0], color="b",ls='--')
        axs[row,column].axvline(hdpi_data_nf_sigma2[dim][1], color="b",ls='--')
        
        
        axs[row,column].axvline(hdpi_data_true_sigma3[dim][0], color="r",ls='-.')
        axs[row,column].axvline(hdpi_data_true_sigma3[dim][1], color="r",ls='-.')
        axs[row,column].axvline(hdpi_data_nf_sigma3[dim][0], color="b",ls='-.')
        axs[row,column].axvline(hdpi_data_nf_sigma3[dim][1], color="b",ls='-.')
        
        
    fig.savefig(path_to_plot,dpi=300)
    fig.clf()

    return
'''
def marginal_plot(target_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_to_plot,ndims):


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
                   r"$a_{0\,\phi}^{A0}$",
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

                ##Get sigmas lists
        hdpi_data_true_sigma1=hdpis_dict_true.get('hdpi_1sigma')
        hdpi_data_nf_sigma1=hdpis_dict_nf.get('hdpi_1sigma')
        hdpi_data_true_sigma2=hdpis_dict_true.get('hdpi_2sigma')
        hdpi_data_nf_sigma2=hdpis_dict_nf.get('hdpi_2sigma')
        hdpi_data_true_sigma3=hdpis_dict_true.get('hdpi_3sigma')
        hdpi_data_nf_sigma3=hdpis_dict_nf.get('hdpi_3sigma')


        print(ndims)

        print('marginal plot')

        n_bins=50
        fig, axs = plt.subplots(int((ndims+1)/10),10, tight_layout=True,squeeze=True)
        
        print(axs.shape)
        for dim in range(ndims-9):
        
            i=dim
            print('hello')
            print(dim)
            row=int(dim/10)
            column=int(dim%10)

            axs[row,column].hist(target_samples[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(nf_samples[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(True)
            
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
            
            
            
            axs[row,column].text(.0,1.13,labels[dim],
            horizontalalignment='left',
            transform=axs[row,column].transAxes,fontsize=22)
            
       
            for j in range(len(hdpi_data_true_sigma1[i])):
           
                axs[row,column].axvline(hdpi_data_true_sigma1[i][j][0], color="r",ls='-')
                axs[row,column].axvline(hdpi_data_true_sigma1[i][j][1], color="r",ls='-')
        
            for j in range(len(hdpi_data_nf_sigma1[i])):
        
                axs[row,column].axvline(hdpi_data_nf_sigma1[i][j][0], color="b",ls='-')
                axs[row,column].axvline(hdpi_data_nf_sigma1[i][j][1], color="b",ls='-')
            
            for j in range(len(hdpi_data_true_sigma2[i])):
   
                axs[row,column].axvline(hdpi_data_true_sigma2[i][j][0], color="r",ls='--')
                axs[row,column].axvline(hdpi_data_true_sigma2[i][j][1], color="r",ls='--')
        
            for j in range(len(hdpi_data_nf_sigma2[i])):
        
                axs[row,column].axvline(hdpi_data_nf_sigma2[i][j][0], color="b",ls='--')
                axs[row,column].axvline(hdpi_data_nf_sigma2[i][j][1], color="b",ls='--')
        
            for j in range(len(hdpi_data_true_sigma3[i])):
           
                axs[row,column].axvline(hdpi_data_true_sigma3[i][j][0], color="r",ls='-.')
                axs[row,column].axvline(hdpi_data_true_sigma3[i][j][1], color="r",ls='-.')
        
            for j in range(len(hdpi_data_nf_sigma3[i])):
        
                axs[row,column].axvline(hdpi_data_nf_sigma3[i][j][0], color="b",ls='-.')
                axs[row,column].axvline(hdpi_data_nf_sigma3[i][j][1], color="b",ls='-.')
            
            axs[row,column].axes.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
            axs[row,column].axes.tick_params(labelsize=17)
            axs[row,column].xaxis.offsetText.set_fontsize(17)
            '''
            axs[row,column].axvline(hdpi_data_true_sigma1[dim][0], color="r",ls='-')
            axs[row,column].axvline(hdpi_data_true_sigma1[dim][1], color="r",ls='-')
            axs[row,column].axvline(hdpi_data_nf_sigma1[dim][0], color="b",ls='-')
            axs[row,column].axvline(hdpi_data_nf_sigma1[dim][1], color="b",ls='-')
        
        
            axs[row,column].axvline(hdpi_data_true_sigma2[dim][0], color="r",ls='--')
            axs[row,column].axvline(hdpi_data_true_sigma2[dim][1], color="r",ls='--')
            axs[row,column].axvline(hdpi_data_nf_sigma2[dim][0], color="b",ls='--')
            axs[row,column].axvline(hdpi_data_nf_sigma2[dim][1], color="b",ls='--')
        
        
            axs[row,column].axvline(hdpi_data_true_sigma3[dim][0], color="r",ls='-.')
            axs[row,column].axvline(hdpi_data_true_sigma3[dim][1], color="r",ls='-.')
            axs[row,column].axvline(hdpi_data_nf_sigma3[dim][0], color="b",ls='-.')
            axs[row,column].axvline(hdpi_data_nf_sigma3[dim][1], color="b",ls='-.')
            '''
        
        for column in range(10):
            i=column
            print(column)
            if column==9:
                axs[8,column].axis('off')
                break
            axs[8,column].hist(target_samples[:,80+column], bins=n_bins,density=True,histtype='step',color='red')
            axs[8,column].hist(nf_samples[:,80+column], bins=n_bins,density=True,histtype='step',color='blue')
        
            x_axis = axs[8,column].axes.get_xaxis()
            x_axis.set_visible(True)
           
            y_axis = axs[8,column].axes.get_yaxis()
            y_axis.set_visible(False)
            
            dim=column+80
            axs[8,column].text(.0,1.13,labels[dim],
            horizontalalignment='left',
            transform=axs[8,column].transAxes,fontsize=22)
            
                       
        
            for j in range(len(hdpi_data_true_sigma1[80+i])):
           
                axs[8,column].axvline(hdpi_data_true_sigma1[80+i][j][0], color="r",ls='-')
                axs[8,column].axvline(hdpi_data_true_sigma1[80+i][j][1], color="r",ls='-')
        
            for j in range(len(hdpi_data_nf_sigma1[80+i])):
        
                axs[8,column].axvline(hdpi_data_nf_sigma1[80+i][j][0], color="b",ls='-')
                axs[8,column].axvline(hdpi_data_nf_sigma1[80+i][j][1], color="b",ls='-')
            
            for j in range(len(hdpi_data_true_sigma2[80+i])):
                print(len(hdpi_data_true_sigma2))
                print(len(hdpi_data_true_sigma2[80+i]))
                axs[8,column].axvline(hdpi_data_true_sigma2[80+i][j][0], color="r",ls='--')
                axs[8,column].axvline(hdpi_data_true_sigma2[80+i][j][1], color="r",ls='--')
        
            for j in range(len(hdpi_data_nf_sigma2[80+i])):
        
                axs[8,column].axvline(hdpi_data_nf_sigma2[80+i][j][0], color="b",ls='--')
                axs[8,column].axvline(hdpi_data_nf_sigma2[80+i][j][1], color="b",ls='--')
        
            for j in range(len(hdpi_data_true_sigma3[80+i])):
           
                axs[8,column].axvline(hdpi_data_true_sigma3[80+i][j][0], color="r",ls='-.')
                axs[8,column].axvline(hdpi_data_true_sigma3[80+i][j][1], color="r",ls='-.')
        
            for j in range(len(hdpi_data_nf_sigma3[80+i])):
        
                axs[8,column].axvline(hdpi_data_nf_sigma3[80+i][j][0], color="b",ls='-.')
                axs[8,column].axvline(hdpi_data_nf_sigma3[80+i][j][1], color="b",ls='-.')
            axs[8,column].axes.ticklabel_format(style='sci',scilimits=(0,0))
            axs[8,column].axes.tick_params(labelsize=17)
            axs[8,column].xaxis.offsetText.set_fontsize(17)
            
            '''
            axs[8,column].axvline(hdpi_data_true_sigma1[80+dim][0], color="r",ls='-')
            axs[8,column].axvline(hdpi_data_true_sigma1[80+dim][1], color="r",ls='-')
            axs[8,column].axvline(hdpi_data_nf_sigma1[dim][0], color="b",ls='-')
            axs[8,column].axvline(hdpi_data_nf_sigma1[dim][1], color="b",ls='-')
        
        
            axs[8,column].axvline(hdpi_data_true_sigma2[dim][0], color="r",ls='--')
            axs[8,column].axvline(hdpi_data_true_sigma2[dim][1], color="r",ls='--')
            axs[8,column].axvline(hdpi_data_nf_sigma2[dim][0], color="b",ls='--')
            axs[8,column].axvline(hdpi_data_nf_sigma2[dim][1], color="b",ls='--')
        
        
            axs[8,column].axvline(hdpi_data_true_sigma3[dim][0], color="r",ls='-.')
            axs[8,column].axvline(hdpi_data_true_sigma3[dim][1], color="r",ls='-.')
            axs[8,column].axvline(hdpi_data_nf_sigma3[dim][0], color="b",ls='-.')
            axs[8,column].axvline(hdpi_data_nf_sigma3[dim][1], color="b",ls='-.')
            '''
        
        blue_line = mlines.Line2D([], [],lw=10, color='red', label='true')
        red_line = mlines.Line2D([], [],lw=10, color='blue', label='pred')
        hdpi1_line = mlines.Line2D([], [],lw=2, color='black',linestyle='-', label='HDPI$_{1\sigma}$')
        hdpi2_line = mlines.Line2D([], [],lw=2, color='black',linestyle='--', label='HDPI$_{2\sigma}$')
        hdpi3_line = mlines.Line2D([], [],lw=2, color='black',linestyle='-.', label='HDPI$_{3\sigma}$')
    
    
    
        fig.legend(handles=[blue_line,red_line,hdpi1_line,hdpi2_line,hdpi3_line],fontsize='xx-large',bbox_to_anchor=[.98,.12])
        fig.set_size_inches(32,21)
        fig.savefig(path_to_plot,dpi=400)
        fig.clf()

        return

def CornerPlotter(target_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_to_plot,selection_list):


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

    #labels=labels[[1,2,3,4,5,19,20,21,22,78,79,80,81,82,83,84,85,86,87,88]]
    print(len(labels))
    labels_3=list( labels[i] for i in selection_list )
    plt.rcParams.update({'font.size': 25})
    
    print(labels_3)
    print(len(labels_3))
    ndims=np.shape(target_samples)[1]

    red_bins=50
    density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
   
    blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    blue_bins=blue_bins.astype(int).tolist()

    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner.corner(target_samples,color='red',bins=red_bins)
    fig=corner.corner(nf_samples,color='blue',bins=blue_bins,fig=figure,max_n_ticks=3)#,labels=labels_3)
    ndims=np.shape(nf_samples)[1]
    
    figure=DrawCIs(figure,hdpis_dict_true,hdpis_dict_nf,ndims,selection_list,labels_3)
    #figure.set_size_inches((30,20))
    blue_line = mlines.Line2D([], [],lw=30, color='red', label='true')
    red_line = mlines.Line2D([], [],lw=30, color='blue', label='pred')
    hdpi1_line = mlines.Line2D([], [],lw=5, color='black',linestyle='-', label='HPDI$_{1\sigma}$')
    hdpi2_line = mlines.Line2D([], [],lw=5, color='black',linestyle='--', label='HPDI$_{2\sigma}$')
    hdpi3_line = mlines.Line2D([], [],lw=5, color='black',linestyle='-.', label='HPDI$_{3\sigma}$')
    
    #fig.set_size_inches(20,16)
    
    plt.legend(handles=[blue_line,red_line,hdpi1_line,hdpi2_line,hdpi3_line], bbox_to_anchor=(-ndims+10.8, ndims+.7, 1., 0.) ,fontsize='xx-large')
    plt.savefig(path_to_plot,bbox_inches='tight',pil_kwargs={'quality':50})
    plt.close()


    return
    
'''
X_data_test_file = '../data/HEPFit_data/Data_new/X_data_test_flavor_new_500K'
true_test_samples=ImportTrueData(X_data_test_file)
true_test_samples=true_test_samples[:500000,:]

print(np.shape(true_test_samples))
path_to_result='/Users/humberto/Documents/work/NFs/FlavorDistribution-2/Best_of_Bests_w_distance/results_ndims_89_bijector_MsplineN_nbijectors_2_splinekonts_16_rangemin_-6_nsamples_100000_batchsize_1024_hiddenlayers_1024-1024-1024_eps_regulariser_0.0001/'

path_to_nf_sample=path_to_result+'/nf_sample_NOTpostprocessed_500ksamples_softclip.npy'
path_to_corner_plot=path_to_result+'/corner_plot_hdpi_wilson_paper_2.png'
nf_samples=ImportNFdata(path_to_nf_sample)
nf_samples=nf_samples[:np.shape(true_test_samples)[0],:]
ndims=np.shape(true_test_samples)[1]
print(ndims)
'''

def save_hpdi_results(output_dir,big_mean_sigma1,big_median_sigma1,big_mean_sigma2,big_median_sigma2,big_mean_sigma3,big_median_sigma3):



        result_frame=pd.read_csv(output_dir+'/results.txt')
        result_frame['mean_hdpi_1sigma']=big_mean_sigma1
        result_frame['mean_hdpi_2sigma']=big_mean_sigma2
        result_frame['mean_hdpi_3sigma']=big_mean_sigma3
        result_frame['median_hdpi_1sigma']=big_median_sigma1
        result_frame['median_hdpi_2sigma']=big_median_sigma2
        result_frame['median_hdpi_3sigma']=big_median_sigma3
        
        result_frame.to_csv(output_dir+'/results_whdpi.txt')

        return


def GeneratePlotandHDPIResults(true_test_samples,nf_samples,path_to_result,ndims):

    path_to_corner_plot=path_to_result+'/corner_plot_hdpi.png'
    path_marginal_plot=path_to_result+'/marginal_plot_hdpi.png'

    hdpi1,hdpi2,hdpi3=HDPI(true_test_samples[:,0])

    print(hdpi1)
    print(hdpi2)
    print(hdpi3)


    bhdpi1,bhdpi2,bhdpi3=HDPI(nf_samples[:,0])

    hdpis_dict_true=HDPIOverDims(true_test_samples)
    hdpis_dict_nf=HDPIOverDims(nf_samples)
    hdpis_dict_difs=ComputeHDPIDif(hdpis_dict_true,hdpis_dict_nf)


    hdpis_dict_rel_difs=ComputeHDPIRelDif(hdpis_dict_true,hdpis_dict_difs)



    big_mean_sigma1,big_median_sigma1,big_mean_sigma2,big_median_sigma2,big_mean_sigma3,big_median_sigma3=GetMeansAndMedians(hdpis_dict_rel_difs)

    save_hpdi_results(path_to_result,big_mean_sigma1,big_median_sigma1,big_mean_sigma2,big_median_sigma2,big_mean_sigma3,big_median_sigma3)

    hdpi_data1_sigma1=hdpis_dict_true.get('hdpi_1sigma')

    marginal_plot(true_test_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_marginal_plot,ndims)

    selection_list=[77,78,79,80,81,82,83,84,85,86,87,88]
    CornerPlotter(true_test_samples[:,selection_list],nf_samples[:,selection_list],hdpis_dict_true,hdpis_dict_nf,path_to_corner_plot,selection_list)
    return
