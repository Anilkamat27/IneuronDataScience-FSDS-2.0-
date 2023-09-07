import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import argparse

def get_data():
    try:
        df = pd.read_csv("wine.csv")
        return df
    except Exception as e:
        raise e

def evaluate(y_true,y_pred,pred_prob):
    '''mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r2=r2_score(y_true,y_pred)'''
    
    accuracy = accuracy_score(y_true,y_pred)
    roc_score = roc_auc_score(y_true,pred_prob, multi_class = 'ovr')
    return accuracy , roc_score
    
def main(n_estimators,max_depth):
    try :
        df= get_data()

        # train test split
        train,test = train_test_split(df)
        X_train = train.drop(["quality"],axis=1)
        X_test = test.drop(["quality"],axis=1)
        y_train= train[["quality"]]
        y_test= test[["quality"]]
        
        with mlflow.start_run():
            # model training 
            rf = RandomForestClassifier(n_estimators = n_estimators,max_depth=max_depth)
            rf.fit(X_train,y_train)
            pred=rf.predict(X_test)
            
            pred_prob = rf.predict_proba(X_test)

            # evaluate model

            accuracy, roc_score = evaluate(y_test,pred,pred_prob)
            mlflow.log_param("n_estimators",n_estimators)
            mlflow.log_param("max_depth",max_depth)
            mlflow.log_metrics({"accuracy": accuracy, "roc_auc_score": roc_score})

            mlflow.sklearn.log_model(rf,"randomforestmodel")
            print(f"accuracy : {accuracy}, roc_score,{roc_score}")

    except Exception as e:
        raise e


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n",default = 50, type = int)
    args.add_argument("--max_depth", "-m",default = 25, type = int)
    parse_args = args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
            raise e   