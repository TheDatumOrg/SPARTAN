import os
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="data/ucr/Univariate_ts/")
    parser.add_argument("-c", "--classifier", required=False, default="spartan")
    parser.add_argument("-g",'--config',required=False,default=None) # config file for methods
    parser.add_argument("-i", "--itr", required=False, default=1)
    parser.add_argument("-n", "--norm", required=False, default="zscore")  # zscore as default
    parser.add_argument("-s","--save_model",required=False, default=None)
    parser.add_argument("-t","--top_num",required=False, default=0, type=int) # test top t datasets
    parser.add_argument("-p","--downsample",default=1.0, type=float)
    parser.add_argument("-e","--eval_task",required=False, default='classification', type=str) # classfication, clustering, tlb, anomaly
    parser.add_argument("-k","--clust_model",required=False, default='kmedoids', type=str)  # -- clustering experiments
    parser.add_argument("-dl","--dataset_list",required=False, default='dataset_list_full', type=str, choices=['test', 'dataset_list1', 'dataset_list2', 'dataset_list3', 'dataset_list_full'])
    parser.add_argument("-l","--linkage",default='complete')
    parser.add_argument("-o","--kmedoids_type",default='pam')
    parser.add_argument("-b","--data_split",default='split', type=str)
    parser.add_argument("-r","--repr_type",default='single', type=str)
    parser.add_argument("--alpha_min", required=False, default=4, type=int)  # -- tlb experiments
    parser.add_argument("--alpha_max", required=False, default=10, type=int) 
    parser.add_argument("--wordlen_min", required=False, default=4, type=int) 
    parser.add_argument("--wordlen_max", required=False, default=10, type=int) 

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    # load arguments
    arguments = parse_arguments()
    data_path = arguments.data
    classifier_name = arguments.classifier
    normalization = arguments.norm
    itr = arguments.itr
    config = arguments.config
    top_num = arguments.top_num
    eval_task = arguments.eval_task
    
    clust_model = arguments.clust_model
    linkage = arguments.linkage
    kmedoids_type = arguments.kmedoids_type
    data_split = arguments.data_split
    repr_type = arguments.repr_type
    downsample_rate = arguments.downsample
    dataset_list = arguments.dataset_list
    
    alphabet_max = arguments.alpha_max
    alphabet_min = arguments.alpha_min
    wordlen_max  = arguments.wordlen_max
    wordlen_min  = arguments.wordlen_min

    # load ucr summary
    dset_info = pd.read_csv('benchmark/util/summaryUnivariate.csv')
    dset_info = dset_info.sort_values(by=['numTrainCases','numTestCases'])

    if eval_task in ['classification', 'clustering', 'tlb']:
        for i in range(dset_info.shape[0]):
            
            # only test first top_num samples (when top_num > 0)
            if top_num - 1 < i and top_num > 0:
                continue

            dataset = dset_info['problem'].iloc[i]

            print("Dataset No.: ", i, dataset)

            if eval_task == 'classification':
                call_string = 'python -m benchmark.eval_classfication --data {} --classifier {} --norm {} --problem {} --itr {} --config {} --downsample {}'.format(data_path,classifier_name,normalization,dataset,itr,config, downsample_rate)
            elif eval_task == 'clustering':

                if i >= dset_info.shape[0] - 2 and repr_type == 'bop': # skip large dataset for oom
                    continue
                call_string = 'python -m benchmark.eval_clustering --data {} --classifier {} --norm {} --problem {} --itr {} --config {} --clust_model {} --linkage {} --kmedoids_type {} -b {} -t {}'.format(data_path,classifier_name,normalization,dataset,itr,config, clust_model, linkage, kmedoids_type, data_split, repr_type)
            elif eval_task == 'tlb':
                call_string = 'python -m benchmark.eval_tlb --data {} --problem {} -x {} --alpha_max {} --alpha_min {} --wordlen_max {} --wordlen_min {}'.format(data_path, dataset, i, alphabet_max, alphabet_min, wordlen_max, wordlen_min)

            os.system(call_string)

    elif eval_task == 'anomaly':

        AD_dataset_list_full = ['Occupancy', 'SensorScope', 'SMD',         'Dodgers',   'Genesis', 'IOPS', 
                             'MGAB',      'NAB',         'NASA-MSL',    'NASA-SMAP', 'Daphnet', 'ECG',
                             'GHL',       'OPPORTUNITY', 'MITDB',       'YAHOO',     'SVDB',    'KDD21'] # change the dataset name as you need
        AD_dataset_list1 = ['Occupancy', 'SensorScope', 'SMD',         'Dodgers',   'Genesis', 'IOPS']
        AD_dataset_list2 = ['MGAB',      'NAB',         'NASA-MSL',    'NASA-SMAP', 'Daphnet',  'ECG']
        AD_dataset_list3 = ['OPPORTUNITY',   'GHL',    'MITDB',       'YAHOO',     'SVDB',    'KDD21']
        AD_dataset_list_test = ['KDD21']

        if dataset_list == 'dataset_list1':
            dataset_list = AD_dataset_list1
        elif dataset_list == 'dataset_list2':
            dataset_list = AD_dataset_list2
        elif dataset_list == 'dataset_list3':
            dataset_list = AD_dataset_list3
        elif dataset_list == 'dataset_list_full':
            dataset_list = AD_dataset_list_full
        elif dataset_list == 'test':
            dataset_list = AD_dataset_list_test
        else:
            raise ValueError(f"Wrong dastaset list {dataset_list} selection.")

        for dataset in dataset_list:

            datadir = os.path.join(data_path,dataset)
            for i, filename in enumerate(sorted(os.listdir(datadir))):
                if not filename.endswith('.out'):
                    continue
                
                # skip extremely large data
                df = pd.read_csv(os.path.join(datadir,filename)).dropna()
                data = df.iloc[:, 0:-1].values.astype(float).reshape(-1)
                if len(data) > 1000000:
                    continue

                # only test first top_num samples (when top_num > 0)
                if top_num - 1 < i and top_num > 0:
                    continue

                call_string = 'python -m benchmark.eval_anomaly --data {} --classifier {} --norm {} --problem {} --itr {} --config {} -t {}'.format(datadir,classifier_name,normalization,filename,itr,config, repr_type)

                os.system(call_string)  
