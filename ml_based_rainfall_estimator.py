'''
Created on 28/set/2017

@author: Massimo Guarascio
@author: Gianluigi Folino
'''
import os
import numpy as np
import pandas as pd
from time import time
from random import seed
import collections
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from mlxtend.classifier.stacking_classification import StackingClassifier
from sklearn.metrics.regression import mean_squared_error
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm.classes import SVR

def readSingleDataFile(path, delim): 
    "READ SINGLE FILE"
    try:
        df=pd.read_csv(path, delim, engine="python", header = 0)
    except ValueError:
        print("Loading error: ", path)
        print(ValueError)
        sys.exit(-1)
    return df

#Discretiziation via predefinied thresholds
def discretize(value, thresholds):
    val=0
    for i in thresholds:
        if value < i:
            return val
        else:
            val=val+1
    return val

def preprocess_rains(rains, data, thresholds):
    for r in rains:
        data[r] = data[r].apply(discretize, thresholds = thresholds)
    return data

    
def compute_macro_per_class(cm, i):
    
    tp = cm[i][i]    
    fn = 0
    fp = 0
    
    for j in range(0,cm.shape[1]):
        if ( j!= i):
            fn += cm[i][j]
            
    for j in range(0,cm.shape[0]):
        if ( j!= i):
            fp += cm[j][i]
    
    csi = float (tp)  / (tp + fp + fn) 
    far = float (fp) / (fp + tp)
    pod = float (tp) / (tp + fn)        
    
    return csi, far, pod

def compute_statistics_macro(cm, rmse):

    num_classes = cm.shape[0]    
    csi=0
    far=0
    pod=0
    
    for i in range(0,num_classes):
    
        measure_per_class = compute_macro_per_class(cm, i)
        csi +=measure_per_class[0]
        far +=measure_per_class[1]
        pod +=measure_per_class[2]

    csi/=num_classes
    far/=num_classes
    pod/=num_classes

    tp_3=0; fp_3=0; fn_3=0; tp_4=0; fp_4=0; fn_4=0;
    
    for i in range(0,cm.shape[0]):
        for j in range(0,cm.shape[1]):
            if i==j:
                if i==3:
                    tp_3 += cm[i][j]
                    continue
                if i==4:
                    tp_4 += cm[i][j]
                    continue
            else:
                if j==3:
                    fp_3 += cm[i][j]
                if j==4:
                    fp_4 += cm[i][j]
                if i==3:
                    fn_3 += cm[i][j]
                if i==4:
                    fn_4 += cm[i][j]
    
    
    precision_class_3 = float (tp_3)/(tp_3+fp_3);  recall_class_3 = float (tp_3)/(tp_3+fn_3); precision_class_4 = float (tp_4)/(tp_4+fp_4); recall_class_4 = float (tp_4)/(tp_4+fn_4);
    f_measure_3 = 2*precision_class_3*recall_class_3/(precision_class_3+recall_class_3)
    f_measure_4 = 2*precision_class_4*recall_class_4/(precision_class_4+recall_class_4)

    if (debug):
        print("precision class 3: ", precision_class_3)
        print("recall class 3: ", recall_class_3)
        print("precision class 4: ", precision_class_4)
        print("recall class 4: ", recall_class_4)
        print("f_measure class 3: ", f_measure_3)
        print("f_measure class 4: ", f_measure_4)
    
    return (csi,far,pod,rmse,precision_class_3,recall_class_3,f_measure_3,precision_class_4,recall_class_4,f_measure_4)

def create_ensemble(s, num_base_learners=5, n_estimators = 10):
    
    seed = s
    base_models = []
    
    i = 1
    for m in range(0, num_base_learners):
        m = RandomForestClassifier(n_estimators=n_estimators, random_state=seed+i, n_jobs=10)
        base_models.append(m)
        i=i+1
        
    meta_classifier = GaussianNB()
    multi_view_stacker = StackingClassifier(base_models, use_probas = True, use_features_in_secondary = False, meta_classifier = meta_classifier)
    return multi_view_stacker

def read_params():
    
    #Read parameter from command line
    if len (sys.argv) != 5:
        print("Usage: %s <dataset path> <training file> <test file> <debug (0=no debug; 1=debug)>" % sys.argv[0])
        sys.exit(-1)
    
    dataset_path = 	sys.argv[1]
    trainingset = sys.argv[2]
    testset = sys.argv[3]
    debug= int(sys.argv[4])
    
    return trainingset, testset, debug, dataset_path


if __name__ == "__main__":
       
    #debug parameter enables verbose printing
     
    #Class name
    class_name = "rain"
    
    trainingset, testset, debug, dataset_path=read_params()
    
    #delim for csv
    input_separator = ";"
    
    #Thresholds for discretizing rain value in classes according IRPI evaluation
    thresholds = [0.5, 2.5, 7.5, 15]

    #CHOOSE algorithm: 0 - DecisionTree; 1 - RandomForest; 2 - Stacker;  3 - ADABoost; 4 - SVR ; 5 - GaussianNB
    #algorithm 2 is our main approach
    alg = 2
    
    #Algorithm parameters
    tree_depth = 7
    random_forest_size = 50
    boosting_size = 50
    seed = 0
    
    #READING DATA
    t0 = time()
    training_set = readSingleDataFile(os.path.join(dataset_path,trainingset), input_separator)
    
    #PRINT Training set CLASS DISTRIBUTION
    if debug:
        print(training_set[class_name].value_counts())
 
    if debug:
        print("loaded in %0.3fs." % (time() - t0))
    
    t0 = time()
    
    test_set = readSingleDataFile(os.path.join(dataset_path,testset), input_separator)
    
    #PRINT Test set CLASS DISTRIBUTION
    if debug:
        print(test_set[class_name].value_counts())
        
    if debug:
        print("loaded in %0.3fs." % (time() - t0))
     
    #PREPROCESSING STEP - handling missing values, one-hot encoding, normalization numeric values, etc.    
    if debug:
        "PREPARING DATA"
    
    training_x = training_set.drop(class_name, 1)
    training_y = training_set[class_name]
    
    test_x = test_set.drop(class_name, 1)
    test_y = test_set[class_name]
    
    if debug:
        print("HANDLING nearest rain gauge records...")
    
    training_x = preprocess_rains(["rain1","rain2","rain3","rain4"], training_x, thresholds)
    test_x = preprocess_rains(["rain1","rain2","rain3","rain4"], test_x, thresholds)
    
    training_x = training_x.drop(["x","y","x1","y1","x2","y2","x3","y3","x4","y4","Met11", "original_rain"], axis=1)
    test_x = test_x.drop(["x","y","x1","y1","x2","y2","x3","y3","x4","y4","Met11", "original_rain"], axis=1)
        
    if debug:
        print("HANDLING MISSING VALUES")
    
    categorical_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    categorical_imp.fit(training_x[["rain1","rain2","rain3","rain4"]])
    training_x[["rain1","rain2","rain3","rain4"]] = categorical_imp.transform(training_x[["rain1","rain2","rain3","rain4"]])
    test_x[["rain1","rain2","rain3","rain4"]] = categorical_imp.transform(test_x[["rain1","rain2","rain3","rain4"]]) 
      
    t0 = time()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(training_x)
    training_x[list(training_x)] = imp.transform(training_x)
    test_x[list(test_x)] = imp.transform(test_x)    
    if (debug):
        print("replace missing values in %0.3fs." % (time() - t0))
         
    training_x_no_missing = training_x
    test_x_no_missing = test_x    
    
    indexes_to_encode = []
    for v in ["rain1","rain2","rain3","rain4"]:
        indexes_to_encode.append(training_x_no_missing.columns.get_loc(v))
    
    if debug: 
        print("indexes to encode ", indexes_to_encode)
    
    #LEARNING STEP
    if debug:
        print("LEARNING STEP")
   
    #default 
    classifier = "not_init"
    
    if alg == 0:
        classifier = DecisionTreeClassifier(max_depth=tree_depth)
    if alg == 1:
        classifier = RandomForestClassifier(n_estimators=random_forest_size, random_state=seed, n_jobs=10) 
    if alg == 2:
        classifier = create_ensemble(seed)
    if alg == 3:
        classifier = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=boosting_size, random_state=seed)
    if alg == 4:
        scaler = StandardScaler()
        svr = SVR(kernel='rbf', cache_size=4000, C=1e3, gamma=0.0001, max_iter=200000, epsilon=0.0001)
        classifier = Pipeline([('standardize',scaler) , ('svr', svr)])
    if alg == 5:
        classifier = GaussianNB()    
    
    if classifier == "not_init":
        print("Classifier not init, exit")
        exit(-1)
    
    if debug:
        print("TRAINING MODEL...")
    
    classifier.fit(training_x_no_missing, training_y)
      
    # EVALUTION 
    if debug:
        print("EVALUATION") 

    pred_y = classifier.predict(test_x_no_missing)
    
    #round to nearest class  
    pred_y = np.around(pred_y, 0)
    
    #Handling SVR result
    if alg == 4 :
        for v in pred_y:
            print(v)
        pred_y = [max(0,k) for k in pred_y]
        pred_y = [min(4,k) for k in pred_y]
        
    if debug:   
        print(str(confusion_matrix(test_y, pred_y)).replace("[", " ").replace("]"," "))
    if debug:    
        print(classification_report(test_y, pred_y)) 
       
    mse = mean_squared_error(test_y, pred_y)
    
    cm = confusion_matrix(test_y, pred_y)
    
    header = "CSI\tFAR\tPOD\tMSE\tPREC_4\tREC_4\tF-M_4\tPREC_5\tREC_5\tF-M_5"
    results = compute_statistics_macro(cm, mse)
    
    print(header)
    for i in results:
        print ("%.3f" % i, '\t', end='')
    print()
    

    

    
    
    
    
    
