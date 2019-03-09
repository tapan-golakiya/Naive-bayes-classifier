import pandas as pd
import numpy as np
import math
def main():
    dataset = pd.read_csv('creditcard.csv')
    print(dataset.shape)
    msk = np.random.rand(len(dataset)) < 0.8
    training_set = dataset[msk]
    testing_set = dataset[~msk]
    naiveBayes(training_set,testing_set)

def naiveBayes(training_set,testing_set):
    var = training_set.groupby(['Class']).size()
    #divide by class 0 and 1
    fraud_no = var[0]
    fraud_yes = var[1]
    total_samples = fraud_yes + fraud_no
    fraud_yes_prob = fraud_yes/total_samples
    fraud_no_prob = fraud_no/total_samples
    
    #divide time into categories and find probabilty for each
    near_time_count = training_set['Time'][training_set['Time'] < 84786].count()
    far_time_count = training_set['Time'][training_set['Time'] >= 84786].count()
    no_fraud_near_time_count = training_set['Time'][(training_set['Time'] < 84786) & (training_set['Class'] == 1)].count()
    fraud_near_time_count = training_set['Time'][(training_set['Time'] < 84786) & (training_set['Class'] == 0)].count()
    no_fraud_far_time_count = training_set['Time'][(training_set['Time'] >= 84786) & (training_set['Class'] == 1)].count()
    fraud_far_time_count = training_set['Time'][(training_set['Time'] >= 84786) & (training_set['Class'] == 0)].count()
    prob_no_fraud_near_time = no_fraud_near_time_count / near_time_count
    prob_no_fraud_far_time = no_fraud_far_time_count / far_time_count
    prob_fraud_near_time = fraud_near_time_count / near_time_count
    prob_fraud_far_time = fraud_far_time_count / far_time_count
    
    #divide amount into categories and find probability for each
    low_amount_count = training_set['Amount'][training_set['Amount'] < 10].count()
    no_fraud_low_amount_count = training_set['Amount'][(training_set['Amount'] < 10) & (training_set['Class'] == 1)].count()
    fraud_low_amount_count = training_set['Amount'][(training_set['Amount'] < 10) & (training_set['Class'] == 0)].count()
    
    med_amount_count = training_set['Amount'][(training_set['Amount'] >= 10) & (training_set['Amount'] <50)].count()
    no_fraud_med_amount_count = training_set['Amount'][(training_set['Amount'] >= 10) & (training_set['Amount'] <50) & (training_set['Class'] == 1)].count()
    fraud_med_amount_count = training_set['Amount'][(training_set['Amount'] >= 10) & (training_set['Amount'] <50) & (training_set['Class'] == 0)].count()
    
    high_amount_count = training_set['Amount'][training_set['Amount']>50].count()
    no_fraud_high_amount_count = training_set['Amount'][(training_set['Amount']>50) & (training_set['Class'] == 1)].count()
    fraud_high_amount_count = training_set['Amount'][(training_set['Amount']>50) & (training_set['Class'] == 0)].count()

    prob_no_fraud_low_amount = no_fraud_low_amount_count / low_amount_count
    prob_fraud_low_amount = fraud_low_amount_count / low_amount_count
    prob_no_fraud_med_amount = no_fraud_med_amount_count / med_amount_count
    prob_fraud_med_amount = fraud_med_amount_count / med_amount_count
    prob_no_fraud_high_amount = no_fraud_high_amount_count / high_amount_count
    prob_fraud_high_amount = fraud_high_amount_count / high_amount_count

    #v1 to v28 mean and variance
    mean_array = training_set.groupby('Class').mean()
    var_array = training_set.groupby('Class').var()
    mean_no_fraud = {}
    mean_fraud = {}
    var_no_fraud = {}
    var_fraud = {}
    col_list = training_set.columns.values
    #adding mean and variance to individual arrays
    for col in col_list:
        if ((col != 'Time') & (col != 'Amount') & (col !='Class')):
            var1 = mean_array[col][mean_array.index == 1].values[0]
            mean_fraud[col] = float(var1)
            var2 = mean_array[col][mean_array.index == 0].values[0]
            mean_no_fraud[col] = float(var2)
            var3 = var_array[col][var_array.index == 1].values[0]
            var_fraud[col] = float(var3)
            var4 = var_array[col][var_array.index == 0].values[0]
            var_no_fraud[col] = float(var4)
            
    testing_samples = testing_set.shape[0]
    predicted_classes = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    
    #############               TESTING           ##################################################33
    for j in range(testing_samples):
        row = testing_set.iloc[[j]]
        if(row.iloc[0]['Time'] < 84786):
            prob_no_fraud_time_2 = prob_no_fraud_near_time
            prob_fraud_time_2 = prob_fraud_near_time
        else:
            prob_no_fraud_time_2 = prob_no_fraud_far_time
            prob_fraud_time_2 = prob_fraud_far_time
            
        if(row.iloc[0]['Amount'] < 10):
            prob_no_fraud_amount_2 = prob_no_fraud_low_amount
            prob_fraud_amount_2 = prob_fraud_low_amount
        elif((row.iloc[0]['Amount'] >= 10) & (row.iloc[0]['Amount'] < 50)):
            prob_no_fraud_amount_2 = prob_no_fraud_med_amount
            prob_fraud_amount_2 = prob_fraud_med_amount
        else:
            prob_no_fraud_amount_2 = prob_no_fraud_high_amount
            prob_fraud_amount_2 = prob_fraud_high_amount
            
        #gaussian probility for v1 to v28 for class 0
        no_prob_v1 = gaussian_probability(row.iloc[0]['V1'],mean_no_fraud['V1'],var_no_fraud['V1'])
        no_prob_v2 = gaussian_probability(row.iloc[0]['V2'],mean_no_fraud['V2'],var_no_fraud['V2'])
        no_prob_v3 = gaussian_probability(row.iloc[0]['V3'],mean_no_fraud['V3'],var_no_fraud['V3'])
        no_prob_v4= gaussian_probability(row.iloc[0]['V4'],mean_no_fraud['V4'],var_no_fraud['V4'])
        no_prob_v5 = gaussian_probability(row.iloc[0]['V5'],mean_no_fraud['V5'],var_no_fraud['V5'])
        no_prob_v6 = gaussian_probability(row.iloc[0]['V6'],mean_no_fraud['V6'],var_no_fraud['V6'])
        no_prob_v7 = gaussian_probability(row.iloc[0]['V7'],mean_no_fraud['V7'],var_no_fraud['V7'])
        no_prob_v8 = gaussian_probability(row.iloc[0]['V8'],mean_no_fraud['V8'],var_no_fraud['V8'])
        no_prob_v9 = gaussian_probability(row.iloc[0]['V9'],mean_no_fraud['V9'],var_no_fraud['V9'])
        no_prob_v10 = gaussian_probability(row.iloc[0]['V10'],mean_no_fraud['V10'],var_no_fraud['V10'])
        no_prob_v11 = gaussian_probability(row.iloc[0]['V11'],mean_no_fraud['V11'],var_no_fraud['V11'])
        no_prob_v12 = gaussian_probability(row.iloc[0]['V12'],mean_no_fraud['V12'],var_no_fraud['V12'])
        no_prob_v13 = gaussian_probability(row.iloc[0]['V13'],mean_no_fraud['V13'],var_no_fraud['V13'])
        no_prob_v14 = gaussian_probability(row.iloc[0]['V14'],mean_no_fraud['V14'],var_no_fraud['V14'])
        no_prob_v15 = gaussian_probability(row.iloc[0]['V15'],mean_no_fraud['V15'],var_no_fraud['V15'])
        no_prob_v16 = gaussian_probability(row.iloc[0]['V16'],mean_no_fraud['V16'],var_no_fraud['V16'])
        no_prob_v17 = gaussian_probability(row.iloc[0]['V17'],mean_no_fraud['V17'],var_no_fraud['V17'])
        no_prob_v18 = gaussian_probability(row.iloc[0]['V18'],mean_no_fraud['V18'],var_no_fraud['V18'])
        no_prob_v19 = gaussian_probability(row.iloc[0]['V19'],mean_no_fraud['V19'],var_no_fraud['V19'])
        no_prob_v20 = gaussian_probability(row.iloc[0]['V20'],mean_no_fraud['V20'],var_no_fraud['V20'])
        no_prob_v21 = gaussian_probability(row.iloc[0]['V21'],mean_no_fraud['V21'],var_no_fraud['V21'])
        no_prob_v22 = gaussian_probability(row.iloc[0]['V22'],mean_no_fraud['V22'],var_no_fraud['V22'])
        no_prob_v23 = gaussian_probability(row.iloc[0]['V23'],mean_no_fraud['V23'],var_no_fraud['V23'])
        no_prob_v24 = gaussian_probability(row.iloc[0]['V24'],mean_no_fraud['V24'],var_no_fraud['V24'])
        no_prob_v25 = gaussian_probability(row.iloc[0]['V25'],mean_no_fraud['V25'],var_no_fraud['V25'])
        no_prob_v26 = gaussian_probability(row.iloc[0]['V26'],mean_no_fraud['V26'],var_no_fraud['V26'])
        no_prob_v27 = gaussian_probability(row.iloc[0]['V27'],mean_no_fraud['V27'],var_no_fraud['V27'])
        no_prob_v28 = gaussian_probability(row.iloc[0]['V28'],mean_no_fraud['V28'],var_no_fraud['V28'])
        
        #gaussian probility for v1 to v28 for class 0
        yes_prob_v1 = gaussian_probability(row.iloc[0]['V1'],mean_fraud['V1'],var_fraud['V1'])
        yes_prob_v2 = gaussian_probability(row.iloc[0]['V2'],mean_fraud['V2'],var_fraud['V2'])
        yes_prob_v3 = gaussian_probability(row.iloc[0]['V3'],mean_fraud['V3'],var_fraud['V3'])
        yes_prob_v4= gaussian_probability(row.iloc[0]['V4'],mean_fraud['V4'],var_fraud['V4'])
        yes_prob_v5 = gaussian_probability(row.iloc[0]['V5'],mean_fraud['V5'],var_fraud['V5'])
        yes_prob_v6 = gaussian_probability(row.iloc[0]['V6'],mean_fraud['V6'],var_fraud['V6'])
        yes_prob_v7 = gaussian_probability(row.iloc[0]['V7'],mean_fraud['V7'],var_fraud['V7'])
        yes_prob_v8 = gaussian_probability(row.iloc[0]['V8'],mean_fraud['V8'],var_fraud['V8'])
        yes_prob_v9 = gaussian_probability(row.iloc[0]['V9'],mean_fraud['V9'],var_fraud['V9'])
        yes_prob_v10 = gaussian_probability(row.iloc[0]['V10'],mean_fraud['V10'],var_fraud['V10'])
        yes_prob_v11 = gaussian_probability(row.iloc[0]['V11'],mean_fraud['V11'],var_fraud['V11'])
        yes_prob_v12 = gaussian_probability(row.iloc[0]['V12'],mean_fraud['V12'],var_fraud['V12'])
        yes_prob_v13 = gaussian_probability(row.iloc[0]['V13'],mean_fraud['V13'],var_fraud['V13'])
        yes_prob_v14 = gaussian_probability(row.iloc[0]['V14'],mean_fraud['V14'],var_fraud['V14'])
        yes_prob_v15 = gaussian_probability(row.iloc[0]['V15'],mean_fraud['V15'],var_fraud['V15'])
        yes_prob_v16 = gaussian_probability(row.iloc[0]['V16'],mean_fraud['V16'],var_fraud['V16'])
        yes_prob_v17 = gaussian_probability(row.iloc[0]['V17'],mean_fraud['V17'],var_fraud['V17'])
        yes_prob_v18 = gaussian_probability(row.iloc[0]['V18'],mean_fraud['V18'],var_fraud['V18'])
        yes_prob_v19 = gaussian_probability(row.iloc[0]['V19'],mean_fraud['V19'],var_fraud['V19'])
        yes_prob_v20 = gaussian_probability(row.iloc[0]['V20'],mean_fraud['V20'],var_fraud['V20'])
        yes_prob_v21 = gaussian_probability(row.iloc[0]['V21'],mean_fraud['V21'],var_fraud['V21'])
        yes_prob_v22 = gaussian_probability(row.iloc[0]['V22'],mean_fraud['V22'],var_fraud['V22'])
        yes_prob_v23 = gaussian_probability(row.iloc[0]['V23'],mean_fraud['V23'],var_fraud['V23'])
        yes_prob_v24 = gaussian_probability(row.iloc[0]['V24'],mean_fraud['V24'],var_fraud['V24'])
        yes_prob_v25 = gaussian_probability(row.iloc[0]['V25'],mean_fraud['V25'],var_fraud['V25'])
        yes_prob_v26 = gaussian_probability(row.iloc[0]['V26'],mean_fraud['V26'],var_fraud['V26'])
        yes_prob_v27 = gaussian_probability(row.iloc[0]['V27'],mean_fraud['V27'],var_fraud['V27'])
        yes_prob_v28 = gaussian_probability(row.iloc[0]['V28'],mean_fraud['V28'],var_fraud['V28'])
        
        no_fraud_prob = prob_no_fraud_time_2 * prob_no_fraud_amount_2 * no_prob_v1 * no_prob_v2 * no_prob_v3 * no_prob_v4 * no_prob_v5 * no_prob_v6 * no_prob_v7 * no_prob_v8 * no_prob_v9 *no_prob_v10 * no_prob_v11 * no_prob_v12 * no_prob_v13 * no_prob_v14 * no_prob_v15 * no_prob_v16 * no_prob_v17 * no_prob_v18 * no_prob_v19 * no_prob_v20 * no_prob_v21 * no_prob_v22 * no_prob_v23 * no_prob_v24 * no_prob_v25 * no_prob_v26 * no_prob_v27 * no_prob_v28 * fraud_no_prob
        yes_fraud_prob = prob_fraud_time_2 * prob_fraud_amount_2 * yes_prob_v1 * yes_prob_v2 * yes_prob_v3 * yes_prob_v4 * yes_prob_v5 * yes_prob_v6 * yes_prob_v7 * yes_prob_v8 * yes_prob_v9 *yes_prob_v10 * yes_prob_v11 * yes_prob_v12 * yes_prob_v13 * yes_prob_v14 * yes_prob_v15 * yes_prob_v16 * yes_prob_v17 * yes_prob_v18 * yes_prob_v19 * yes_prob_v20 * yes_prob_v21 * yes_prob_v22 * yes_prob_v23 * yes_prob_v24 * yes_prob_v25 * yes_prob_v26 * yes_prob_v27 * yes_prob_v28 * fraud_yes_prob
        
        if(yes_fraud_prob > no_fraud_prob):
            result = 1
        else:
            result = 0
        
        if((result == 1) & (row.iloc[0]['Class'] == 1)):
            tp = tp + 1
        elif((result == 1) & (row.iloc[0]['Class'] == 0)):
            fp = fp + 1
        elif((result == 0) & (row.iloc[0]['Class'] == 1)):
            fn = fn + 1
        else:
            tn = tn + 1
        
        predicted_classes.append(result)
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (recall * precision)/(recall + precision)

        
def gaussian_probability(n, mean, sd):
    x = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*x)**.5
    num = math.exp(-(float(n)-float(mean))**2/(2*x))
    return num/denom

if __name__ == '__main__':
    main()
        
    