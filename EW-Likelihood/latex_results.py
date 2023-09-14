import pandas as pd
import math

def ReadResultsFrame(path_to_frame):

    results_frame=pd.read_csv(path_to_frame)

    
    return results_frame
    



def OpenFile(path):

    txtfile=open(path+'/latex_table_new_metrics.txt','w')
    return txtfile


def FirstLines(txtfile):

    return txtfile
    
def LastLines(txtfile):

    return txtfile


def NextLine(txtfile,labels,results_frame,j):

    next_line=labels[j]+' & '+str(truncate(results_frame['ks_test'][j]))+' & '+str(truncate(results_frame['mean_hdpi_1sigma'][j]))+' & '+str(truncate(results_frame['mean_hdpi_2sigma'][j]))+' & '+str(truncate(results_frame['mean_hdpi_3sigma'][j]))+' \\'
    
    txtfile.write(next_line)
    txtfile.write('\n')
    txtfile.write('\midrule')
    txtfile.write('\n')

    return txtfile
def CloseFile(txtfile):

    txtfile.close()
    
    return
    
def truncate(number):
    "truncate float to 3 decimal places"
        
    if number<1 and number>1e-90:
            
        provisional_number=number
        order=0
    
        while provisional_number<1:
        
            provisional_number=provisional_number*(10)
            order=order+1
            
        
        factor=10**3
        truncated=(math.trunc(provisional_number * factor) / (factor*10**(order)))
   
    #elif number>1e-90:
    # truncated=0.0
    
    else:
        factor=10**4
        truncated=math.trunc(number * factor) / factor
    return truncated

def LatexMain(path_frame_main_results):
    main_results_frame=pd.read_csv(path_frame_main_results)
    txtfile_main=open(path+'/latex_table_main_new_metrics.txt','w')
    Main_line=str(truncate(main_results_frame['KS_pv_mean'][0]))+' & '+str(truncate(main_results_frame['SWD_mean'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_1sigma'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_2sigma'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_3sigma'][0]))+' \\'


    txtfile_main.write(Main_line)
    txtfile_main.close()
    return

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
                   r'$c_{\varphi l}^{1}$',
                   r'$c_{\varphi l}^{3}$',
                   r'$c_{\varphi q}^{1}$',
                   r'$c_{\varphi q}^{3}$',
                   r'$c_{\varphi d}$',
                   r'$c_{\varphi e}$',
                   r'$c_{\varphi u}$',
                   r'$c_{ll}$',
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


coeff_list=[14,15,16,17,18,19,20,21]
path='./results/official_results/'
path_to_frame=path+'/results_new_metrics__200000_all_whdpi.txt'

path_frame_main_results=path+'/results_new_metrics__200000_whdpi.txt'


LatexMain(path_frame_main_results)
results_frame=ReadResultsFrame(path_to_frame)
txtfile=OpenFile(path)


for j in coeff_list:
    NextLine(txtfile,labels,results_frame,j)
