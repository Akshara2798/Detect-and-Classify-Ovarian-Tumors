# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager


def calculate_specificity(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    specificity = tn / (tn + fp)
    return specificity

def k_fold(y_true, y_pred, k_values):
    for k in k_values:
        k = min(k, len(y_true))
        if len(y_true) >= k:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            kappa_scores = []
            specificity_scores = []
            mses=[]
            rmses=[]
            mapes=[]
            maes=[]
            for train_index, test_index in kf.split(y_true):
                y_true_train, y_true_test = y_true[train_index], y_true[test_index]
                y_pred_train, y_pred_test = y_pred[train_index], y_pred[test_index]
                accuracy = accuracy_score(y_true_test, y_pred_test)
                precision = precision_score(y_true_test, y_pred_test, average='macro')
                recall = recall_score(y_true_test, y_pred_test, average='macro')
                f1 = f1_score(y_true_test, y_pred_test, average='macro')
                kappa = cohen_kappa_score(y_true_test, y_pred_test)
                specificity = calculate_specificity(y_true_test, y_pred_test)
                mse=mean_squared_error(y_true_test,y_pred_test)
                rmse = np.sqrt(np.mean((y_true_test - y_pred_test) ** 2))
                mape = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
                accuracy_scores.append(accuracy)
                mae=mean_absolute_error(y_true_test, y_pred_test)
                maes.append(mae)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                kappa_scores.append(kappa)
                specificity_scores.append(specificity)
                mses.append(mse)
                rmses.append(rmse)
                mapes.append(mape)
                
            mean_accuracy = np.mean(accuracy_scores)
            mean_precision = np.mean(precision_scores)
            mean_recall = np.mean(recall_scores)
            mean_f1 = np.mean(f1_scores)
            mean_kappa = np.mean(kappa_scores)
            mean_specificity=np.mean(specificity_scores)
            mean_mse=np.mean(mses)
            mean_mape=np.mean(mapes)
            mean_rmse=np.mean(rmses)
        else:
            print(f"\n{k}-fold Cross-Validation: Not enough samples for k-fold with k={k}.")

    return accuracy_scores, precision_scores, recall_scores, f1_scores,specificity_scores, mses,rmses,maes

    
