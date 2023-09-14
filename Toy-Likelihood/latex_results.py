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
labels = ['\mu']


coeff_list=[0]
path='./results/newmet_1/run_2_b/'

path_frame_main_results=path+'/results_new_metrics__200000_whdpi.txt'
LatexMain(path_frame_main_results)

path_to_frame=path+'/results_new_metrics__200000_all_whdpi.txt'
results_frame=ReadResultsFrame(path_to_frame)
txtfile=OpenFile(path)

for j in coeff_list:
    NextLine(txtfile,labels,results_frame,j)
