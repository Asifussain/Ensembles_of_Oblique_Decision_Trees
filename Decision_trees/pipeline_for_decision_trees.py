#
#.............Pipeline for calling the Oblique Decision Trees using the Scikit-Learn's Bagging Classifier...............
#
#
# Importing all the oblique decision trees
#
#

from WODT import *
from HouseHolder_CART import *
from RandCART import *
from CO2 import *
from NDT import *
from Oblique_Classifier_1 import *
from DNDT import *
from segmentor import *
#
#
# Importing all the packages
#
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier



def pre_process(dataset='breastcancer'):
    X, y = None, None

    if dataset == 'abalone':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/abalone/abalone_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/abalone/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'acute-inflammation':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/acute-inflammation/acute-inflammation_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/acute-inflammation/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'acute-nephritis':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/acute-nephritis/acute-nephritis_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/acute-nephritis/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'annealing':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/annealing/annealing_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/annealing/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'audiology-std':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/audiology-std/audiology-std_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/audiology-std/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'balance-scale':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/balance-scale/balance-scale_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/balance-scale/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'balloons':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/balloons/balloons_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/balloons/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'blood':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/blood/blood_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/blood/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'breast-cancer':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/breast-cancer/breast-cancer_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/breast-cancer/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()

    elif dataset == 'breast-cancer-wisc':
        X_data = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/breast-cancer-wisc/breast-cancer-wisc_py.dat', header=None, delimiter=',')
        X = np.array(X_data)
        y = pd.read_csv('Ensembles_of_Oblique_Decision_Trees/dataset/breast-cancer-wisc/labels_py.dat', header=None, delimiter=',')
        y = np.array(y, dtype=int).ravel()


    
    else:
        ValueError('Unknown results set!')

    return X, y



def make_estimator(method='wodt', max_depth=5, n_estimators=10):
    if method == 'wodt':
        return WeightedObliqueDecisionTreeClassifier(max_depth=max_depth)
    elif method == 'oc1':
        return ObliqueClassifier1(max_depth=max_depth)
    elif method == 'stdt':
        return DecisionTreeClassifier(max_depth=max_depth)
    elif method == 'ndt':
        return NDTClassifier(max_depth=max_depth)
    elif method == 'wodt_bag':
        return BaggingClassifier(base_estimator=WeightedObliqueDecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'oc1_bag':
        return BaggingClassifier(base_estimator=ObliqueClassifier1(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'stdt_bag':
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'ndt_bag':
        return BaggingClassifier(base_estimator=NDTClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'hhcart':
        return HHCartClassifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'randcart':
        return RandCARTClassifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'co2':
        return CO2Classifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'hhcart_bag':
        return BaggingClassifier(base_estimator=HHCartClassifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'randcart_bag':
        return BaggingClassifier(base_estimator=RandCARTClassifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'co2_bag':
        return BaggingClassifier(base_estimator=CO2Classifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'random_forest':
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    else:
        ValueError('Unknown model!')


def evaluate(datasets_to_evaluate, methods_to_evaluate):
    n_depths = 10
    n_estimators = 5
    
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(index=datasets_to_evaluate, columns=methods_to_evaluate)
    
    for dataset in datasets_to_evaluate:
        print('\n--- Evaluating results set: {0}'.format(dataset))
        X, y = pre_process(dataset)
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.33, random_state=42)
        for method in methods_to_evaluate:
                if method == 'dndt_bag':

                    transformed_Y = np.reshape(train_Y, (-1, 1))
                    onehotencoder = OneHotEncoder()
                    transformed_Y = onehotencoder.fit_transform(transformed_Y).toarray()
                    d = X.shape[1]
                    num_class = len(np.unique(y))
                    Y_pred = dndt_fit(train_X, test_X, transformed_Y, d, num_class, n_estimators)
                    acc = np.mean(Y_pred == test_Y)

                else:

                    estimator = make_estimator(method=method, max_depth=n_depths, n_estimators=n_estimators)
                    estimator.fit(train_X, train_Y)
                    Y_pred = estimator.predict(test_X)
                    acc = np.mean(Y_pred == test_Y)
                    
                    # Store the accuracy in the DataFrame
                results_df.loc[dataset, method] = acc
    
    return results_df



if __name__ == '__main__':
    
    datasets = ['abalone','acute-inflammation','acute-nephritis','annealing','audiology-std','balance-scale','balloons','blood','breast-cancer','breast-cancer-wisc']
    methods = ['hhcart_bag','stdt_bag','randcart_bag',]
    results =  evaluate(datasets, methods)


