
import gc
from inspect import stack
import json
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Any, List, Optional, Union, Dict
from lightgbm import LGBMClassifier
import numpy as np  # type:ignore
import optuna
import pandas as pd  # type:ignore
import wandb
from lightgbm import log_evaluation
from loguru import logger
from pydantic import BaseModel
import sklearn
from .training_config import TrainingConfig
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler            
from sklearn.impute import KNNImputer
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from feature_engine.encoding import WoEEncoder
from sklearn.linear_model import HuberRegressor
from catboost import CatBoostClassifier

class TPSTrainer(BaseModel):
    config: TrainingConfig
    train: Optional[pd.DataFrame]
    test: Optional[pd.DataFrame]
    target: Optional[np.ndarray]
    train_features: Optional[np.ndarray]
    test_features: Optional[np.ndarray]
    model: Any
    y_pred_list: Optional[np.ndarray] = []
    current_time: int = time()
    wandb_params_table: Any
    params_suggested: Optional[Dict]
    params_notsuggested: Optional[Dict]
    stack_classifier:Any

    class Config:
        arbitrary_types_allowed = True

    
    def setup(self):
        """Run pre-training tasks (set up data, logger)"""

        self.train = pd.read_csv(self.config.train_path)
        self.test = pd.read_csv(self.config.test_path)
        self.target = self.train.failure
       
        cat_features = ['product_code', 'attribute_0', 'attribute_1']
        for categorical_feature in cat_features:
    
            le_encoder = LabelEncoder()
        
            
            le_encoder.fit(self.train[categorical_feature])
            self.test[categorical_feature]  = self.test[categorical_feature].map(lambda s: '<unknown>' if s not in le_encoder.classes_ else s)
            le_encoder.classes_ = np.append(le_encoder.classes_, '<unknown>')
            self.train[categorical_feature]  = le_encoder.transform(self.train[categorical_feature] )
            self.test[categorical_feature]  = le_encoder.transform(self.test[categorical_feature])

        features = [feat for feat in self.train.columns if feat.startswith('measurement')]
        frames = []
    
        self.train = self.impute_data(self.train)
        self.test = self.impute_data(self.test)

        

        # self.train.fillna(self.train.mean(), inplace=True)
        # self.test.fillna(self.test.mean(), inplace=True)

         
        self.train_features = [
            f for f in self.train.columns if f != "id" and f != "failure"
        ]
        self.test_features = [
            f for f in self.test.columns if f != "id" and f != "failure"
        ]
        logger.debug(
            f"Shapes: target:{self.target.shape}, train:{self.train.shape}, test:{self.test.shape}"
        )      # standardizing test data

        sc = StandardScaler() 

        self.train[self.train_features] = sc.fit_transform(self.train[self.train_features])   # standardizing training data
        self.test[self.test_features] = sc.transform(self.test[self.test_features] )      
        
      
        logger.debug("Data setup complete")
        return self

    def create_submission(self):
        
        print(self.test.index)
        print(len(self.y_pred_list))
        subm = pd.DataFrame(
            {

                "id": self.test.id,
                "failure": self.y_pred_list
            }
        )
        Path(self.config.submission_path).mkdir(parents=True, exist_ok=True)
        print(subm.head())
        copy = subm['failure']
        q1,q2,q3,q4 = copy.quantile(0.04),copy.quantile(0.3),copy.quantile(0.7),copy.quantile(0.95)
        u1 = []
        l1 = []
        u2 = []
        l2 = []
        for i in range(len(copy)):
            u1.append(copy[i]>=q3)
            l1.append(copy[i]<=q2)
            u2.append(copy[i]>=q4)
            l2.append(copy[i]<=q1)
        prediction = copy.copy()
        prediction = prediction.apply(lambda x:x*1.1 if x>=q3 else x)
        prediction = prediction.apply(lambda x:x*0.9 if x<=q2 else x)
        prediction[u2] = 1
        prediction =  prediction.apply(lambda x:1 if x>1 else x)
        prediction[l2] = 0
        print(prediction)
        subm['failure'] = np.round(prediction,3)
        subm.to_csv(
            Path(self.config.submission_path) / "submission.csv",
            index=False,
        )
        logger.debug("Submission file generated")

    def impute_data(self,data):
        features = [feat for feat in self.train.columns if feat.startswith('measurement') or feat=='loading']
        frames = []
    
        for code in data.product_code.unique():
            df = data[data.product_code==code].copy()
            imputer = KNNImputer(n_neighbors=3)
            imputer.fit(df[features])
            df[features] = imputer.transform(df[features])
            frames.append(df)
        
        data =  pd.concat(frames)    
        data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
        data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
        data['area'] = data['attribute_2'] * data['attribute_3']
        data['missing(3*5)'] = data['m5_missing'] * (data['m3_missing'])
        return data

    def run_kfold(self, is_test: bool = False) -> List[float]:
        """Fit model to data and return score"""
        score_list, y_pred_list = [], []
        kf = StratifiedKFold(n_splits=self.config.n_splits)

        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            X_tr, X_va, y_tr, y_va = (
                None,
                None,
                None,
                None,
            )  # TODO: is this line necessry?
            X_tr = self.train.iloc[idx_tr][self.train_features]
            X_va = self.train.iloc[idx_va][self.train_features]
            y_tr = self.target[idx_tr]
            y_va = self.target[idx_va]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="roc_auc_score",
                    callbacks=[log_evaluation(1000)],
                )
            X_tr, y_tr = None, None
            y_va_pred = self.model.predict_proba(X_va)[:, 1]
            score = metrics.roc_auc_score(y_va, y_va_pred)
            n_trees = self.model.best_iteration_
            if n_trees is None:
                n_trees = self.model.n_estimators
            logger.debug(f"Fold {fold}, {n_trees:5} trees, Score = {score:.5f}")
            score_list.append(score)

       
        return score_list
    
    def kf_cross_val(self,model,X,y):
    
        scores,feature_imp, features = [],[], []
        
        kf = KFold(n_splits=5,shuffle = True, random_state=42)
        
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            
            x_train = X.iloc[train_index]
            y_train = y.loc[train_index]
            x_test = X.loc[test_index]
            y_test = y.loc[test_index]
            
            model.fit(x_train,y_train)
            
            y_pred = model.predict_proba(x_test)[:,1]     # edit 
            scores.append(metrics.roc_auc_score(y_test,y_pred))
            
            try:
                feature_imp.append(model.feature_importances_)
                features.append(model.feature_names_)
            except AttributeError: # if model does not have .feature_importances_ attribute
                pass
            
        return feature_imp, scores, features

    def objective(self, trial: Any):
        """Optuna objective"""
        self.model = self.config.model(**self.config.hyperparams(trial=trial))
        score_list = self.run_kfold()
        logger.debug(f"Trial params: {trial.params}")
       

        return np.mean(score_list)

    def model_train(self):
        
       
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: self.objective(trial)
        study.optimize(objective, n_trials=self.config.n_trials)

        logger.debug(f"Number of finished trials: {len(study.trials)}")
        logger.debug(f"Best trial value: {study.best_trial.value}")
        logger.debug(
            f"Params: {json.dumps(study.best_trial.params, indent=2, sort_keys=True)}"
        )
        gc.collect()

        self.model = self.config.model(**study.best_trial.params)
        score_list = self.run_kfold(is_test=True)
        logger.debug(f"score_list: {score_list}")
    
        return self.model
    
    def model_stack_fn(self):
            params = {"max_iter": 200, "C": 0.0001, "penalty": "l2", "solver": "newton-cg"}   # initialising KNeighbors Classifier 

            models = {
                 'catboost':CatBoostClassifier(verbose = 0),
                 'lgbm':self.model_train(),
                  'lr':LogisticRegression(**params),
                }
            results  = {}


            for name,model in models.items():
                
                feature_imp,result,features = self.kf_cross_val(model, self.train[self.train_features], self.target)
                results[name] = result

            for name, result in results.items():
                print("----------\n" + name)
                print(np.mean(result))
                print(np.std(result))
                print(feature_imp)
            
            weights = {'catboost':0.01,
                'lr':0.50,
                'lgbm':0.49,
                }
            for name,model in models.items():
                 model.fit(self.train[self.train_features], self.target)    
            
            preds  = {}

            for name,model in models.items():
                print(name)
                pred = pd.DataFrame(model.predict_proba(self.test[self.test_features])).iloc[:,1]  # second column
                preds[name] = pred
            print(preds)
            y_pred  = np.zeros(self.test.shape[0])
            for name,pred in  preds.items():
                y_pred = y_pred + weights[name] * pred
            self.y_pred_list = y_pred
            return self
    