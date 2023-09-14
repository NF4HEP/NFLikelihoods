import numpy as np

from scipy import stats
from statistics import mean,median
import tensorflow as tf
import GenerativeModelsMetrics as GMetrics






def MeansAndStdMetrics(metric_result):

    metric_mean=np.mean(metric_result)
    print(metric_mean)
    metric_std=np.std(metric_result)

    return metric_mean,metric_std


def SaveNewMetrics(metrics_dict,metric_name,results_path):

    name=results_path+'/'+metric_name+'_dict_sf'
    np.save(name,metrics_dict)


    return


def SaveResultsCurrentNewMetrics(results_dict_newmetrics,path_to_results):

    
    Frame=pd.DataFrame(results_dict_newmetrics)
    current_row= Frame.tail(1)
    current_row.to_csv(path_to_results+'/results_new_metrics_sf.txt',index=False)

    return


def GetMetrics(dist_1,dist_2):




    
    n_samples=np.shape(dist_2)[0]
    print(n_samples)
    TwoSampleTestInputs_tf = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                      dist_2_input = dist_2,
                                                      niter = 100,
                                                      batch_size = int(n_samples/100),
                                                      dtype_input = tf.float64,
                                                      seed_input = 0,
                                                      use_tf = True,
                                                      verbose = True)
                                                      
                                                      
    KSTest_tf = GMetrics.KSTest(TwoSampleTestInputs_tf,
                                                                progress_bar = True,
                                                                verbose = True)
    SWDMetric_tf = GMetrics.SWDMetric(TwoSampleTestInputs_tf,
                                                                progress_bar = True,
                                                                verbose = True)
    KSTest_tf.Test_np()
    SWDMetric_tf.Test_np()


    print(KSTest_tf)
    print(KSTest_tf.Results)
    KSres =KSTest_tf.Results[-1].result_value
    SWDres =SWDMetric_tf.Results[-1].result_value
    return KSres,SWDres

    
