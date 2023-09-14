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


    hdpi1 = az.hdi(posterior_samples, hdi_prob=0.68)
    hdpi2 = az.hdi(posterior_samples, hdi_prob=0.954)
    hdpi3 = az.hdi(posterior_samples, hdi_prob=0.997)

    return list(hdpi1),list(hdpi2),list(hdpi3)


def HDPIOverDims(full_dim_samples):

    ndims=np.shape(full_dim_samples)[1]
    
    hdpis_dict={'hdpi_1sigma':[],'hdpi_2sigma':[],'hdpi_3sigma':[]}
    
    
    
    for k in range(ndims):
        hdpi1,hdpi2,hdpi3=HDPI(full_dim_samples[:,k])
        
        hdpis_dict.get('hdpi_1sigma').append(hdpi1)
        hdpis_dict.get('hdpi_2sigma').append(hdpi2)
        hdpis_dict.get('hdpi_3sigma').append(hdpi3)
    
    
        
        


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
        
        
        
        if k==0:
            print('abs difs')
            print(hdpi_dif1)
      
        
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
        hdpis_dict_rel_difs.get('mean_hdpi_1sigma').append(mean(hdpi_rel_dif1))
        
        
        if k==0:
            print('rel difs')
            print(hdpi_rel_dif1)
            print(mean(hdpi_rel_dif1))
            
        
        
        hdpi_rel_dif2=[abs(b_i/a_i) for a_i, b_i in zip(hdpi_data1_sigma2[k], hdpi_difs_sigma2[k])]
        hdpis_dict_rel_difs.get('hdpi_2sigma').append(hdpi_rel_dif2)
        hdpis_dict_rel_difs.get('mean_hdpi_2sigma').append(mean(hdpi_rel_dif2))
        
        hdpi_rel_dif3=[abs(b_i/a_i) for a_i, b_i in zip(hdpi_data1_sigma3[k], hdpi_difs_sigma3[k])]
        hdpis_dict_rel_difs.get('hdpi_3sigma').append(hdpi_rel_dif3)
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
    


def DrawCIs(figure,hdpis_dict_true,hdpis_dict_nf,ndims,selection_list,labels):


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
        print('hehrweogehgoweg')
        ax = axes[i, i]
        ax.set_title(labels[i],loc='left')
        print(hdpi_data_true_sigma1[i])
        for j in range(len(hdpi_data_true_sigma1_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma1_sel[i][0], color="r",ls='-')
            ax.axvline(hdpi_data_true_sigma1_sel[i][1], color="r",ls='-')
        
        for j in range(len(hdpi_data_nf_sigma1_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma1_sel[i][0], color="b",ls='-')
            ax.axvline(hdpi_data_nf_sigma1_sel[i][1], color="b",ls='-')
            
        for j in range(len(hdpi_data_true_sigma2_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma2_sel[i][0], color="r",ls='--')
            ax.axvline(hdpi_data_true_sigma2_sel[i][1], color="r",ls='--')
        
        for j in range(len(hdpi_data_nf_sigma2_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma2_sel[i][0], color="b",ls='--')
            ax.axvline(hdpi_data_nf_sigma2_sel[i][1], color="b",ls='--')
        
        for j in range(len(hdpi_data_true_sigma3_sel[i])):
           
            ax.axvline(hdpi_data_true_sigma3_sel[i][0], color="r",ls='-.')
            ax.axvline(hdpi_data_true_sigma3_sel[i][1], color="r",ls='-.')
        
        for j in range(len(hdpi_data_nf_sigma3_sel[i])):
        
            ax.axvline(hdpi_data_nf_sigma3_sel[i][0], color="b",ls='-.')
            ax.axvline(hdpi_data_nf_sigma3_sel[i][1], color="b",ls='-.')
        


    return figure


def marginal_plot(target_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_to_plot,ndims):

    labels = [r'$\alpha_{S}(M_{Z})$',
                   r'$\Delta\alpha_{\rm{had}}^{(5)}(M_{Z})$',
                   r'$M_{Z}$',
                   r'$m_{H}$',
                   r'$m_{t}$',
                   r'$\delta_{\Gamma_{Z}}$',
                   r'$\delta_{M_{W}}$',
                   r'$\delta_{R^{0}_{b}}$',
                   r'$\delta_{R^{0}_{c}}$',
                   r'$\delta_{R^{0}_{l}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{b}}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{l}}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{q}}}$',
                   r'$\delta_{\sigma^{0}_{h}}$',
                   r'$C_{\varphi l}^{1}$',
                   r'$C_{\varphi l}^{3}$',
                   r'$C_{\varphi q}^{1}$',
                   r'$C_{\varphi q}^{3}$',
                   r'$C_{\varphi d}$',
                   r'$C_{\varphi e}$',
                   r'$C_{\varphi u}$',
                   r'$C_{ll}$',
                   r'$P_{\tau}^{\rm{pol}}$',
                   r'$M_{W}$',
                   r'$\Gamma_{W}$',
                   r'$\rm{BR}_{W\to l\overline{\nu}_{l}}$',
                   r'$\mathcal{A}_{s}$',
                   r'$R_{uc}$',
                   r'$\sin^{2}\theta_{\rm{eff}}$',
                   r'$\Gamma_{Z}$',
                   r'$\sigma^{0}_{h}$',
                   r'$R^{0}_{l}$',
                   r'$A_{\rm{FB}}^{0,l}$',
                   r'$\mathcal{A}_{l}$',
                   r'$R^{0}_{b}$',
                   r'$R^{0}_{c}$',
                   r'$A_{\rm{FB}}^{0,b}$',
                   r'$A_{\rm{FB}}^{0,c}$',
                   r'$\mathcal{A}_{b}$',
                   r'$\mathcal{A}_{c}$'
]

    plt.rcParams.update({'font.size': 22})
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



def CornerPlotter(target_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_to_plot,selection_list):


    labels = [r'$\alpha_{S}(M_{Z})$',
                   r'$\Delta\alpha_{\rm{had}}^{(5)}(M_{Z})$',
                   r'$M_{Z}$',
                   r'$m_{H}$',
                   r'$m_{t}$',
                   r'$\delta_{\Gamma_{Z}}$',
                   r'$\delta_{M_{W}}$',
                   r'$\delta_{R^{0}_{b}}$',
                   r'$\delta_{R^{0}_{c}}$',
                   r'$\delta_{R^{0}_{l}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{b}}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{l}}}$',
                   r'$\delta_{\sin^{2}\theta_{\rm{eff}^{q}}}$',
                   r'$\delta_{\sigma^{0}_{h}}$',
                   r'$C_{\varphi l}^{1}$',
                   r'$C_{\varphi l}^{3}$',
                   r'$C_{\varphi q}^{1}$',
                   r'$C_{\varphi q}^{3}$',
                   r'$C_{\varphi d}$',
                   r'$C_{\varphi e}$',
                   r'$C_{\varphi u}$',
                   r'$C_{ll}$',
                   r'$P_{\tau}^{\rm{pol}}$',
                   r'$M_{W}$',
                   r'$\Gamma_{W}$',
                   r'$\rm{BR}_{W\to l\overline{\nu}_{l}}$',
                   r'$\mathcal{A}_{s}$',
                   r'$R_{uc}$',
                   r'$\sin^{2}\theta_{\rm{eff}}$',
                   r'$\Gamma_{Z}$',
                   r'$\sigma^{0}_{h}$',
                   r'$R^{0}_{l}$',
                   r'$A_{\rm{FB}}^{0,l}$',
                   r'$\mathcal{A}_{l}$',
                   r'$R^{0}_{b}$',
                   r'$R^{0}_{c}$',
                   r'$A_{\rm{FB}}^{0,b}$',
                   r'$A_{\rm{FB}}^{0,c}$',
                   r'$\mathcal{A}_{b}$',
                   r'$\mathcal{A}_{c}$'
]
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
    
    corner.corner(nf_samples,color='blue',bins=blue_bins,fig=figure,max_n_ticks=3)  # , levels=(0.68,95.,99.7))
    
    ndims=np.shape(nf_samples)[1]
    
    figure=DrawCIs(figure,hdpis_dict_true,hdpis_dict_nf,ndims,selection_list,labels)
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

'''
X_data_test_file = '../data/X_data_test_EW_2_300k_1'
true_test_samples=ImportTrueData(X_data_test_file)
true_test_samples=true_test_samples[:100000,:]

print(np.shape(true_test_samples))
path_to_result='results/run_mspline_4_2_lr001/results_ndims_40_bijector_MsplineN_nbijectors_2_splinekonts_6_rangemin_-6_nsamples_10000_batchsize_512_hiddenlayers_128-128-128_eps_regulariser_0/'
path_to_nf_sample=path_to_result+'/sample_nf.pcl'


nf_samples=ImportNFdata(path_to_nf_sample)
nf_samples=nf_samples[:np.shape(true_test_samples)[0],:]
ndims=np.shape(true_test_samples)[1]
print(ndims)
'''



def GeneratePlotandHDPIResults(true_test_samples,nf_samples,path_to_result,ndims):

    path_to_corner_plot=path_to_result+'/corner_plot_hdpi_paper.png'
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

    #path_marginal_plot=path_to_result+'/marginal_plot_hdpi_paper.png'
    #marginal_plot(true_test_samples,nf_samples,hdpis_dict_true,hdpis_dict_nf,path_marginal_plot,ndims)
    print('hheeeeyyy')
    selection_list=[14,15,16,17,18,19,20,21,10,11,38,39]
    print('ypppuuuuu')
    CornerPlotter(true_test_samples[:,selection_list],nf_samples[:,selection_list],hdpis_dict_true,hdpis_dict_nf,path_to_corner_plot,selection_list)

    print('whatss aaappp')

