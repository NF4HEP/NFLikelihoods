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

def LatexMain(path_frame_main_results):
    main_results_frame=pd.read_csv(path_frame_main_results)
    txtfile_main=open(path+'/latex_table_main_new_metrics.txt','w')
    Main_line=str(truncate(main_results_frame['KS_pv_mean'][0]))+' & '+str(truncate(main_results_frame['SWD_mean'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_1sigma'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_2sigma'][0]))+' & '+str(truncate(main_results_frame['median_hdpi_3sigma'][0]))+' \\'


    txtfile_main.write(Main_line)
    txtfile_main.close()
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


coeff_list=[77,78,79,80,81,82,83,84,85,86,87,88]
path='results/official_results/'
path_to_frame=path+'/results_new_metrics__2000_all_whdpi.txt'
path_frame_main_results=path+'/results_new_metrics__2000_whdpi.txt'


LatexMain(path_frame_main_results)
results_frame=ReadResultsFrame(path_to_frame)
txtfile=OpenFile(path)

for j in coeff_list:
    NextLine(txtfile,labels,results_frame,j)
