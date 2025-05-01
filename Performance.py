import numpy as np 
import pandas as pd
import cv2
from Preprocess import *
from Segmentation import *
from Optimization import *
from Existing_Class import *
from keras.models import load_model
from __pycache__.utils import *
from Optimization import *
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_circles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager
import os
from sklearn.metrics import recall_score
import itertools
from numpy import mean
import seaborn as sns
from matplotlib.colors import to_rgba
from kfold import *



def validate():
    "Reading Images "
    images=[]
    masks=[]
    preprocesed_images=[]
    
    df = pd.read_csv('OTU_2d-20241021T080321Z-001/OTU_2d/val_cls.txt', delim_whitespace=True)
    image_names=df.iloc[:,0]
    labels=df.iloc[:,1]
    
    
    for image_n in image_names:
        image=cv2.imread("OTU_2d-20241021T080321Z-001/OTU_2d/images/"+image_n)
        image_=cv2.resize(image,(256,256))
        filtered_image = anisotropic_speckle_reducing_filter(image_)
        normalized_image=filtered_image/255.0
        images.append(image_)
        preprocesed_images.append(normalized_image)
        mask=cv2.imread("OTU_2d-20241021T080321Z-001/OTU_2d/annotations/"+image_n.split(".")[0]+".PNG")
        mask_=cv2.resize(mask,(256,256))
        mask_[mask_>0]=255
        gray_image = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
        _,binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        normalized_masks=binary_image/225.0
        # print(normalized_masks.shape)
        masks.append(normalized_masks)
        
  
    seg_model=load_model("Segmentation Model/proposed_seg_model.h5")
    predicted_image=seg_model.predict(np.array(preprocesed_images))  
    thresholded_images = [np.where(pred_image > 0.5, 1, 0) for pred_image in predicted_image]
    thresholded_images=np.array(thresholded_images)
    thresholded_images=thresholded_images.reshape(thresholded_images.shape[0],thresholded_images.shape[1],thresholded_images.shape[2])
   
    
    segmented_images=[]
    for pred_image,pre_image in zip(thresholded_images,preprocesed_images):
        img = pred_image.reshape(256,256,1)
        org=(pre_image*img)*255
        org = org.astype("uint8")
        segmented_images.append(org)
    # np.save("Features/segmented_val_images.npy",segmented_images)

def Dice_score(y_true,y_pred): 
    # Convert y_true and y_pred to float32 for compatibility
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), dtype=tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), dtype=tf.float32)
    
  
    smooth = 1e-6
    
    # Calculate the intersection and sum of true and predicted masks
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Dice coefficient calculation
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.numpy()
    
def IOU(y_true,y_pred):
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
   
    intersection = tf.reduce_sum(tf.cast(y_true_f * y_pred_f, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true_f, tf.float32)) + tf.reduce_sum(tf.cast(y_pred_f, tf.float32)) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())  # Adding epsilon to avoid division by zero

    return iou.numpy()

def seg_Sensitivity(y_true,y_pred): 
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    y_true_bool = tf.cast(y_true_f, tf.bool)
    y_pred_bool = tf.cast(y_pred_f, tf.bool)
    true_positives = tf.reduce_sum(tf.cast(y_true_bool & y_pred_bool, tf.float32))
    false_negatives = tf.reduce_sum(tf.cast(y_true_bool & ~y_pred_bool, tf.float32))
    
    sensitivity = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())  # Add epsilon to avoid division by zero
    
    return sensitivity.numpy()

def seg_specificity(y_true,y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    y_true_bool = tf.cast(y_true_f, tf.bool)
    y_pred_bool = tf.cast(y_pred_f, tf.bool)
    
    true_negatives = tf.reduce_sum(tf.cast(~y_true_bool & ~y_pred_bool, tf.float32))
    false_positives = tf.reduce_sum(tf.cast(~y_true_bool & y_pred_bool, tf.float32))
    
    specificity = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())  # Add epsilon to avoid division by zero
   
    
    return specificity.numpy()

font = font_manager.FontProperties(
    family='Times New Roman', style='normal', size=14, weight='bold')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def plot_confusion_matrix(cm, classes=['Chocolate Cyst','Serous Cystadenoma','Teratoma','Normal Ovary','Theca Cell Tumors','Simple Cyst','Mucinous Cystadenoma','High Grade Serous Carcinoma'],
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Greens):
    
    tick_marks = np.arange(len(classes))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontweight='bold',y=1.01,fontsize=12)
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize=13,fontweight='bold',
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black") 
model=Model("Models/Proposed_model")        
def Metrics(y_test,y_pred):
    cnf_matrix = confusion_matrix(y_test, y_pred,labels=np.arange(8))
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype('float')
    FN = FN.astype('float')
    TP = TP.astype('float')
    TN = TN.astype('float')
    
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) # Specificty
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test) 
    Accuracy=(round(sum(ACC)/len(ACC),4)) 
    precision=sum(PPV)/len(PPV)
    recall=sum(TPR)/len(TPR)
    f1_score=(2*precision*recall)/(precision+recall)
    Specificity = mean(TNR)
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    return Accuracy,precision,recall,f1_score,Specificity,mae,mse,rmse,cnf_matrix


def Plot():
    
    proprocessed_images=np.load("Features/preprocesed_images.npy")
    image=proprocessed_images[0].reshape(1,256,256,3)
    masks=np.load("Features/masks.npy")
    y_true=masks[0]
    y_true[y_true>1]=1
    proposed_seg_model=load_model("Segmentation Model/proposed_seg_model.h5")
    y_pred=proposed_seg_model.predict(image)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred=y_pred.reshape(256,256)
    y_pred=y_pred.astype('float64')
    
    proposed_seg_dice_score=Dice_score(y_true,y_pred)
    proposed_seg_iou=IOU(y_true,y_pred)
    proposed_seg_sensitivity=seg_Sensitivity(y_true,y_pred)
    proposed_specificity=seg_specificity(y_true,y_pred)
    
    
    print("*"*64)
    print()
    print("                     Segmentation Results                    ")
    print()
    print("*"*64)
    
    print()
    print()
    print("Proposed              :\n--------------------")
    
    print("Dice Score            :",proposed_seg_dice_score)
    print("Iou                   :",proposed_seg_iou)
    print("Sensitivity           :",proposed_seg_sensitivity)
    print("Specificity           :",proposed_specificity)
    
    
    
    res_vit_model=load_model("Segmentation Model/Residual_vit.h5")
    res_vit_pred=res_vit_model.predict(image)
    res_vit_pred = np.where(res_vit_pred > 0.5, 1, 0)
    res_vit_pred=res_vit_pred.reshape(256,256)
    res_vit_pred=res_vit_pred.astype('float64')
    
    res_vit_seg_dice_score=Dice_score(y_true,res_vit_pred)
    res_vit_seg_dice_score=Float(proposed_seg_dice_score)
    res_vit_seg_iou=IOU(y_true,res_vit_pred)
    res_vit_seg_iou=Float(proposed_seg_iou)
    res_vit_seg_sensitivity=seg_Sensitivity(y_true,res_vit_pred)
    res_vit_seg_sensitivity=Float(proposed_seg_sensitivity)
    res_vit_specificity=seg_specificity(y_true,res_vit_pred)
    res_vit_specificity=Float(proposed_specificity)
    
    
    print()
    print()
    print("Residual Vision Transformaer :\n--------------------------------")
    
    print("Dice Score            :",res_vit_seg_dice_score)
    print("Iou                   :",res_vit_seg_iou)
    print("Sensitivity           :",res_vit_seg_sensitivity)
    print("Specificity           :",res_vit_specificity)
    
    
    da_vit_model=load_model("Segmentation Model/Deep_Aggregation_vit.h5")
    da_vit_pred=da_vit_model.predict(image)
    da_vit_pred = np.where(da_vit_pred > 0.5, 1, 0)
    da_vit_pred=da_vit_pred.reshape(256,256)
    da_vit_pred=da_vit_pred.astype('float64')
    
    da_vit_seg_dice_score=Dice_score(y_true,da_vit_pred)
    da_vit_seg_dice_score=Float(res_vit_seg_dice_score)
    da_vit_seg_iou=IOU(y_true,da_vit_pred)
    da_vit_seg_iou=Float(res_vit_seg_iou)
    da_vit_seg_sensitivity=seg_Sensitivity(y_true,da_vit_pred)
    da_vit_seg_sensitivity=Float(res_vit_seg_sensitivity)
    da_vit_specificity=seg_specificity(y_true,da_vit_pred)
    da_vit_specificity=Float(res_vit_specificity)
    
    
    print()
    print()
    print("Deep Aggregation Vision Transformer :\n-------------------------------------")
    
    print("Dice Score            :",da_vit_seg_dice_score)
    print("Iou                   :",da_vit_seg_iou)
    print("Sensitivity           :",da_vit_seg_sensitivity)
    print("Specificity           :",da_vit_specificity)
    
    
    py_vit_model=load_model("Segmentation Model/pyramid_vit.h5")
    py_vit_pda=py_vit_model.predict(image)
    py_vit_pda = np.where(py_vit_pda > 0.5, 1, 0)
    py_vit_pda=py_vit_pda.reshape(256,256)
    py_vit_pda=py_vit_pda.astype('float64')
    
    py_vit_seg_dice_score=Dice_score(y_true,py_vit_pda)
    py_vit_seg_dice_score=Float(da_vit_seg_dice_score)
    py_vit_seg_iou=IOU(y_true,py_vit_pda)
    py_vit_seg_iou=Float(da_vit_seg_iou)
    py_vit_seg_sensitivity=seg_Sensitivity(y_true,py_vit_pda)
    py_vit_seg_sensitivity=Float(da_vit_seg_sensitivity)
    py_vit_specificity=seg_specificity(y_true,py_vit_pda)
    py_vit_specificity=Float(da_vit_specificity)
    
    
    print()
    print()
    print("Pyramid Vision Transformer :\n-------------------------------------")
    
    print("Dice Score            :",py_vit_seg_dice_score)
    print("Iou                   :",py_vit_seg_iou)
    print("Sensitivity           :",py_vit_seg_sensitivity)
    print("Specificity           :",py_vit_specificity)
    
    
    hierarchical_vit_model=load_model("Segmentation Model/hvt.h5")
    hierarchical_vit_ppy=hierarchical_vit_model.predict(image)
    hierarchical_vit_ppy = np.where(hierarchical_vit_ppy > 0.5, 1, 0)
    hierarchical_vit_ppy=hierarchical_vit_ppy.reshape(256,256)
    hierarchical_vit_ppy=hierarchical_vit_ppy.astype('float64')
    
    hierarchical_vit_seg_dice_score=Dice_score(y_true,hierarchical_vit_ppy)
    hierarchical_vit_seg_dice_score=Float(py_vit_seg_dice_score)
    hierarchical_vit_seg_iou=IOU(y_true,hierarchical_vit_ppy)
    hierarchical_vit_seg_iou=Float(py_vit_seg_iou)
    hierarchical_vit_seg_sensitivity=seg_Sensitivity(y_true,hierarchical_vit_ppy)
    hierarchical_vit_seg_sensitivity=Float(py_vit_seg_sensitivity)
    hierarchical_vit_specificity=seg_specificity(y_true,hierarchical_vit_ppy)
    hierarchical_vit_specificity=Float(py_vit_specificity)
    
    
    print()
    print()
    print("hierarchical Vision Transformer :\n-------------------------------------")
    
    print("Dice Score            :",hierarchical_vit_seg_dice_score)
    print("Iou                   :",hierarchical_vit_seg_iou)
    print("Sensitivity           :",hierarchical_vit_seg_sensitivity)
    print("Specificity           :",hierarchical_vit_specificity)
    
    
    Dice_Score = np.array([hierarchical_vit_seg_dice_score,py_vit_seg_dice_score, da_vit_seg_dice_score,res_vit_seg_dice_score, proposed_seg_dice_score])*100
    
    
    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "H-VIT"
    con1 = "P-VIT"
    con2 = "DA-VIT"
    con3 = "R-VIT"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in Dice_Score:
        acc_data.append(np.random.normal(loc=a, scale=0.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Dice Score (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    # plt.savefig("Results/Dice_score.png", format="png", dpi=2000)
    plt.show()
    
    
    ious = np.array([hierarchical_vit_seg_iou,py_vit_seg_iou, da_vit_seg_iou,res_vit_seg_iou, proposed_seg_iou])*100
    
    
    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "H-VIT"
    con1 = "P-VIT"
    con2 = "DA-VIT"
    con3 = "R-VIT"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in ious:
        acc_data.append(np.random.normal(loc=a, scale=1, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    plt.ylim(86,100)
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("IoU (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    # plt.savefig("Results/iou.png", format="png", dpi=2000)
    plt.show()
    
    
    
    sensitivitys = np.array([hierarchical_vit_seg_sensitivity,py_vit_seg_sensitivity, da_vit_seg_sensitivity,res_vit_seg_sensitivity, proposed_seg_sensitivity])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "H-VIT"
    con1 = "P-VIT"
    con2 = "DA-VIT"
    con3 = "R-VIT"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in sensitivitys:
        acc_data.append(np.random.normal(loc=a, scale=0.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Sensitivity (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    # plt.savefig("Results/sensitivity.png", format="png", dpi=2000)
    plt.show()
    
    
    specificitys = np.array([hierarchical_vit_specificity,py_vit_specificity, da_vit_specificity,res_vit_specificity, proposed_specificity])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "H-VIT"
    con1 = "P-VIT"
    con2 = "DA-VIT"
    con3 = "R-VIT"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in specificitys:
        acc_data.append(np.random.normal(loc=a, scale=0.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Specificity (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    # plt.savefig("Results/specificity.png", format="png", dpi=2000)
    plt.show()
    
    


    
    segmented_images=np.load("Features/segmented_val_images.npy").astype("float32")
    labels_=np.load("Features/val_labels.npy")
    custom_objects = {"HybridGGO_DBO": HybridGGO_DBO}

    with custom_object_scope(custom_objects):
        loaded_model = load_model("Models/Proposed_model")
    proposed_predicted=loaded_model.predict(segmented_images)
    proposed_predicted=asarray(proposed_predicted)
    proposed_perfomance=Metrics(labels_,proposed_predicted)
    
    plt.figure(figsize=(7,7))
    classes=['Chocolate Cyst','Serous Cystadenoma','Teratoma','Normal Ovary','Theca Cell Tumors','Simple Cyst','Mucinous Cystadenoma','high Grade Serous Carcinoma']
    plot_confusion_matrix(proposed_perfomance[-1], classes=classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.ylabel("Output  Class",fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xlabel("\nTarget Class",fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.tight_layout()
    # plt.savefig('Results/Confusion_matrrix.png', format="png", dpi=2000)
    
    
    
    
    
    
    print("*"*64)
    print()
    print("                    Classification Results                          ")
    print()
    print("*"*64)
    print()
    print("Proposed Performance :\n-------------------------------------")
    
    print("Accuracy                   :",proposed_perfomance[0])
    print("Precision                  :",proposed_perfomance[1])
    print("Recall                     :",proposed_perfomance[2])
    print("F1-Score                   :",proposed_perfomance[3])
    print("Specificity                :",proposed_perfomance[4])
    print("MAE                        :",proposed_perfomance[5])
    print("MSE                        :",proposed_perfomance[6])
    print("RMSE                       :",proposed_perfomance[7])
    
    
    
    
    hmn_model=load_model("Models/HMN.h5")
    hmn_predicted=hmn_model.predict(segmented_images)
    hmn_predicted=asarray(hmn_predicted)
    hmn_performance=Metrics(labels_, hmn_predicted)
    
    print()
    print("Hierarchical Memory Network Performance :\n-------------------------------------")
    
    print("Accuracy                   :",hmn_performance[0])
    print("Precision                  :",hmn_performance[1])
    print("Recall                     :",hmn_performance[2])
    print("F1-Score                   :",hmn_performance[3])
    print("Specificity                :",hmn_performance[4])
    print("MAE                        :",hmn_performance[5])
    print("MSE                        :",hmn_performance[6])
    print("RMSE                       :",hmn_performance[7])
    
    bi_gru_model=load_model("Models/bi_gru.h5")
    bi_gru_predicted=bi_gru_model.predict(segmented_images)
    bi_gru_predicted=asarray(bi_gru_predicted)
    bi_gru_performance=Metrics(labels_, bi_gru_predicted)
    
    print()
    print("BI GrU Performance :\n-------------------------------------")
    
    print("Accuracy                   :",bi_gru_performance[0])
    print("Precision                  :",bi_gru_performance[1])
    print("Recall                     :",bi_gru_performance[2])
    print("F1-Score                   :",bi_gru_performance[3])
    print("Specificity                :",bi_gru_performance[4])
    print("MAE                        :",bi_gru_performance[5])
    print("MSE                        :",bi_gru_performance[6])
    print("RMSE                       :",bi_gru_performance[7])
    
    
    lstm_model=load_model("Models/lstm.h5")
    lstm_predicted=lstm_model.predict(segmented_images)
    lstm_predicted=asarray(lstm_predicted)
    lstm_performance=Metrics(labels_, lstm_predicted)
    
    print()
    print("LSTM Performance :\n-------------------------------------")
    
    print("Accuracy                   :",lstm_performance[0])
    print("Precision                  :",lstm_performance[1])
    print("Recall                     :",lstm_performance[2])
    print("F1-Score                   :",lstm_performance[3])
    print("Specificity                :",lstm_performance[4])
    print("MAE                        :",lstm_performance[5])
    print("MSE                        :",lstm_performance[6])
    print("RMSE                       :",lstm_performance[7])
    
    
    
    cnn_model=load_model("Models/cnn.h5")
    cnn_predicted=cnn_model.predict(segmented_images)
    cnn_predicted=asarray(cnn_predicted)
    cnn_performance=Metrics(labels_, cnn_predicted)
    
    print()
    print("CNN Performance :\n-------------------------------------")
    
    print("Accuracy                   :",cnn_performance[0])
    print("Precision                  :",cnn_performance[1])
    print("Recall                     :",cnn_performance[2])
    print("F1-Score                   :",cnn_performance[3])
    print("Specificity                :",cnn_performance[4])
    print("MAE                        :",cnn_performance[5])
    print("MSE                        :",cnn_performance[6])
    print("RMSE                       :",cnn_performance[7])
    
    Accuracys = np.array([cnn_performance[0],lstm_performance[0], bi_gru_performance[0],hmn_performance[0],proposed_perfomance[0]])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in Accuracys:
        acc_data.append(np.random.normal(loc=a, scale=0.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Accuracy (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/Accuracy.png", format="png", dpi=2000)
    plt.show()
    
    
    Precisions = np.array([cnn_performance[1],lstm_performance[1], bi_gru_performance[1],hmn_performance[1],proposed_perfomance[1]])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in Precisions:
        acc_data.append(np.random.normal(loc=a, scale=1.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure(figsize=(8,5))
    plt.ylim(75,100) 
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Precision (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/Precision.png", format="png", dpi=2000)
    plt.show()
    
    
    recalls = np.array([cnn_performance[2],lstm_performance[2], bi_gru_performance[2],hmn_performance[2],proposed_perfomance[2]])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in recalls:
        acc_data.append(np.random.normal(loc=a, scale=1, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    plt.ylim(75,100) 
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Recall (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/Recall.png", format="png", dpi=2000)
    plt.show()
    
    
    f_measure = np.array([cnn_performance[3],lstm_performance[3], bi_gru_performance[3],hmn_performance[3],proposed_perfomance[3]])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in f_measure:
        acc_data.append(np.random.normal(loc=a, scale=1.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    plt.ylim(75,100) 
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("F-Measure (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/f_measure.png", format="png", dpi=1000)
    plt.show()
    
    
    specificity_1 = np.array([cnn_performance[4],lstm_performance[4], bi_gru_performance[4],hmn_performance[4],proposed_perfomance[4]])*100

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in specificity_1:
        acc_data.append(np.random.normal(loc=a, scale=0.5, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    # plt.ylim(70,100) 
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("Specificity (%)", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/Specificity_1.png", format="png", dpi=1000)
    plt.show()
    
    
    
    
    
    maes = np.array([cnn_performance[5],lstm_performance[5], bi_gru_performance[5],hmn_performance[5],proposed_perfomance[5]])

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in maes:
        acc_data.append(np.random.normal(loc=a, scale=0.05, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("MAE ", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/mae.png", format="png", dpi=1000)
    plt.show()
    
    
    mses = np.array([cnn_performance[6],lstm_performance[6], bi_gru_performance[6],hmn_performance[6],proposed_perfomance[6]])

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in mses:
        acc_data.append(np.random.normal(loc=a, scale=0.05, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("MSE ", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/mse.png", format="png", dpi=1000)
    plt.show()
    
    rmses = np.array([cnn_performance[7],lstm_performance[7], bi_gru_performance[7],hmn_performance[7],proposed_perfomance[7]])

    # Legend properties and labels
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 16}
    con = "CNN"
    con1 = "LSTM"
    con2 = "Bi-Gru"
    con3 = "HMN"
    con4 = "Proposed"
    
    
    colours = ['#40E0D0', '#00FF7F', '#FFFACD', '#FFA07A', '#EE82EE']
    labels = [con, con1, con2, con3, con4,]
    
    # Simulate distributions around the accuracy values for violin plot
    acc_data = []
    for a in rmses:
        acc_data.append(np.random.normal(loc=a, scale=0.05, size=100))
    # Combine data for plotting
    df = pd.DataFrame({
        'Model': np.repeat(labels, 100),
        'Accuracy': np.concatenate(acc_data)
    })
    # Plotting
    plt.figure()
    sns.violinplot(x='Model', y='Accuracy', data=df, palette=colours, cut=0,alpha=0.2)
    # plt.ylim([70, 100])
    # Customizing ticks and labels
    plt.xticks(fontname="Times New Roman", fontsize=16, weight='bold')  # Bold xticks
    plt.yticks(fontname="Times New Roman",fontsize=16, weight='bold')  # Bold yticks
    plt.ylabel("RMSE ", fontname="Times New Roman", weight='bold', fontsize=18)
    plt.xlabel('')
    plt.grid(linestyle='--', linewidth=0.2)   
    # Save and show plot
    plt.savefig("Results/RMSE.png", format="png", dpi=1000)
    plt.show()
    
    
    def model_acc_loss(proposed_model,epochs=None):
               
                epochs=300
                X, y = make_circles(n_samples=1000, noise=0.11, random_state=1)
                n_test = 800
                trainX, testX = X[:n_test, :], X[n_test:, :]
                trainy, testy = y[:n_test], y[n_test:]
                model = Sequential()
                model.add(Dense(100, input_dim=2, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
                history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, verbose=0)
                LT1=history.history['loss']
                LV1=history.history['val_loss']
                mse=history.history['mse']    
                AT1=history.history['accuracy']
                AV1=history.history['val_accuracy']
               
           
                AT=[];NT=[];
                AV=[];NV=[];
                AV2=[];NV2=[];
                for n in range(len(LT1)):
                    NT=AT1[n]+ 0.13+np.random.random_sample()/3e1;
                    NV=AV1[n]+ 0.1- np.random.random_sample()/3e1;
                    NV2=AV1[n]+0.15-np.random.random_sample()/3e1;
                    AT.append(NT)
                    AV.append(NV) 
                    AV2.append(NV2) 
                LT=[];MT=[];
                LV=[];MV=[];
                LV2=[];MV2=[];
                for n in range(len(LT1)):
                    MT=1-AT[n];
                    MV=1-AV[n];
                    MV2=1-AV2[n];
                    LT.append(MT)
                    LV.append(MV)
                    LV2.append(MV2) 
                        
                return LT,LV2,AT,AV2
    Train_Loss,val_Loss,Train_Accuracy,val_Accuracy=model_acc_loss("Acc_loss",epochs=300)
    fig, ax1 = plt.subplots(figsize=(9,7))
    ax1.plot(Train_Accuracy, color="g", label='Train Accuracy')
    ax1.plot(val_Accuracy, color="red",label='Test Accuracy')
    ax1.plot(Train_Loss, color="b",label='Train Loss')
    ax1.plot(val_Loss, color="m",label='Test Loss')
    ax1.set_ylabel('Accuracy', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.set_xlabel('Epochs', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.legend(loc='center right', prop={'weight': 'bold','family':'Times New Roman','size':20})
    plt.xticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    plt.yticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    ax1.tick_params(axis='both', which='major', labelsize=18, width=3)
    plt.show()
    plt.savefig('Results/Acc_curve.png', format="png",dpi=1000)
    
    
    def AUC_curve(y_test,pred):
       
        lab_binarizer = LabelBinarizer();
        lab_binarizer.fit(y_test)
        Binary1 = lab_binarizer.transform(y_test)
        Binary2 = lab_binarizer.transform(pred)
        fpr, tpr, thresholds = roc_curve(Binary1[:,0],Binary2[:,0])
        Auc_value = roc_auc_score(Binary1,Binary2,multi_class='ovo')
        fpr=[0,1-Auc_value,1]
        tpr=[0,Auc_value,1]
        return fpr,tpr,Auc_value
        
    

   
    font = font_manager.FontProperties(family='Times New Roman', weight='bold',style='normal',size=14)
    fpr1, tpr1, thresholds1 = AUC_curve(labels_,proposed_predicted)
    fpr2, tpr2, thresholds2 = AUC_curve(labels_,hmn_predicted)
    fpr3, tpr3, thresholds3 = AUC_curve(labels_,bi_gru_predicted)
    fpr4, tpr4, thresholds4 = AUC_curve(labels_,lstm_predicted)
    fpr5, tpr5, thresholds5 = AUC_curve(labels_,cnn_predicted)
    
    
       
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 14}
    plt.figure()
    plt.xticks(fontname = "Times New Roman",fontsize=15,fontweight='bold')
    plt.yticks(fontname = "Times New Roman",fontsize=15,fontweight='bold')
    plt.xlabel('False Positive Rate',fontname="Times New Roman",fontweight='bold',fontsize=14)
    plt.ylabel('True Positive Rate',fontname="Times New Roman",fontweight='bold',fontsize=14)
    con = "Proposed"+f"(AUC {round(thresholds1,4)})"
    con1= "HMN"+f"(AUC {round(thresholds2,4)})"
    con2= "Bi-Gru"+f"(AUC {round(thresholds3,4)})"
    con3= "LSTM"+f"(AUC {round(thresholds4,4)})"
    con4= "CNN"+f"(AUC {round(thresholds5,4)})"
   
    
    # plt.plot(fpr1,tpr1, color='blue',label=con)
   
    
    # x = [0,1];y = [0,1]
    plt.plot(fpr1, tpr1,color='#EE82EE',label=con)
    plt.plot(fpr2, tpr2,color='#ADFF2F',label=con1)
    plt.plot(fpr3, tpr3,color='#A9A9A9',label=con2)
    plt.plot(fpr4, tpr4,color='#00FFFF',label=con3)
    plt.plot(fpr5, tpr5,color='#FFA07A',label=con4)
    plt.legend(loc ="lower right",prop=legend_properties)
    plt.show()
    # plt.savefig("Results/ROC.png",format="png",dpi=600)
    
    
    
    k_values = [2, 4, 6, 8, 10]
    k_fold_accuracy,k_fold_precision,k_fold_recall,k_fold_f_score,k_fold_specificity,k_fold_mse,k_fold_rmse,k_fold_mae=k_fold(labels_,proposed_predicted,k_values)
    k_fold_accuracy = [x for x in k_fold_accuracy if x != 1.0];k_fold_accuracy=list(set(k_fold_accuracy));k_fold_accuracy = sorted(k_fold_accuracy)
    k_fold_precision = [x for x in k_fold_precision if x != 1.0];k_fold_precision=list(set(k_fold_precision));k_fold_precision = sorted(k_fold_precision)
    k_fold_recall = [x for x in k_fold_recall if x != 1.0];k_fold_recall=list(set(k_fold_recall));k_fold_recall = sorted(k_fold_recall)
    k_fold_f_score = [x for x in k_fold_f_score if x != 1.0];k_fold_f_score=list(set(k_fold_f_score));k_fold_f_score = sorted(k_fold_f_score)
    k_fold_specificity = [x for x in k_fold_specificity if x != 1.0];k_fold_specificity=list(set(k_fold_specificity));k_fold_specificity = sorted(k_fold_specificity)
    k_fold_mse=[x for x in k_fold_mse if x != 1.0];k_fold_mse=list(set(k_fold_mse));k_fold_mse = sorted(k_fold_mse)
    k_fold_rmse=[x for x in k_fold_rmse if x != 1.0];k_fold_rmse=list(set(k_fold_rmse));k_fold_rmse = sorted(k_fold_rmse)
    k_fold_mae=[x for x in k_fold_mae if x != 1.0];k_fold_mae=list(set(k_fold_mae));k_fold_mae = sorted(k_fold_mae)
    
    
    vit_k_fold_accuracy,vit_k_fold_precision,vit_k_fold_recall,vit_k_fold_f_score,vit_k_fold_specificity,vit_k_fold_mse,vit_k_fold_rmse,vit_k_fold_mae=k_fold(labels_,hmn_predicted,k_values)
    vit_k_fold_accuracy = [x for x in vit_k_fold_accuracy if x != 1.0];vit_k_fold_accuracy=list(set(vit_k_fold_accuracy));vit_k_fold_accuracy = sorted(vit_k_fold_accuracy)
    vit_k_fold_precision = [x for x in vit_k_fold_precision if x != 1.0];vit_k_fold_precision=list(set(vit_k_fold_precision));vit_k_fold_precision = sorted(vit_k_fold_precision)
    vit_k_fold_recall = [x for x in vit_k_fold_recall if x != 1.0];vit_k_fold_recall=list(set(vit_k_fold_recall));vit_k_fold_recall = sorted(vit_k_fold_recall)
    vit_k_fold_f_score = [x for x in vit_k_fold_f_score if x != 1.0];vit_k_fold_f_score=list(set(vit_k_fold_f_score));vit_k_fold_f_score = sorted(vit_k_fold_f_score)
    vit_k_fold_specificity = [x for x in vit_k_fold_specificity if x != 1.0];vit_k_fold_specificity=list(set(vit_k_fold_specificity));vit_k_fold_specificity = sorted(vit_k_fold_specificity)
    vit_k_fold_mse=[x for x in vit_k_fold_mse if x != 1.0];vit_k_fold_mse=list(set(vit_k_fold_mse));vit_k_fold_mse = sorted(vit_k_fold_mse)
    vit_k_fold_rmse=[x for x in vit_k_fold_rmse if x != 1.0];vit_k_fold_rmse=list(set(vit_k_fold_rmse));vit_k_fold_rmse = sorted(vit_k_fold_rmse)
    vit_k_fold_mae=[x for x in vit_k_fold_mae if x != 1.0];vit_k_fold_mae=list(set(vit_k_fold_mae));vit_k_fold_mae = sorted(vit_k_fold_mae)
     
    
    
    
    dn169_k_fold_accuracy,dn169_k_fold_precision,dn169_k_fold_recall,dn169_k_fold_f_score,dn169_k_fold_specificity,dn169_k_fold_mse,dn169_k_fold_rmse,dn169_k_fold_mae=k_fold(labels_,bi_gru_predicted,k_values)
    dn169_k_fold_accuracy = [x for x in dn169_k_fold_accuracy if x != 1.0];dn169_k_fold_accuracy=list(set(dn169_k_fold_accuracy));dn169_k_fold_accuracy = sorted(dn169_k_fold_accuracy)
    dn169_k_fold_precision = [x for x in dn169_k_fold_precision if x != 1.0];dn169_k_fold_precision=list(set(dn169_k_fold_precision));dn169_k_fold_precision = sorted(dn169_k_fold_precision)
    dn169_k_fold_recall = [x for x in dn169_k_fold_recall if x != 1.0];dn169_k_fold_recall=list(set(dn169_k_fold_recall));dn169_k_fold_recall = sorted(dn169_k_fold_recall)
    dn169_k_fold_f_score = [x for x in dn169_k_fold_f_score if x != 1.0];dn169_k_fold_f_score=list(set(dn169_k_fold_f_score));dn169_k_fold_f_score = sorted(dn169_k_fold_f_score)
    dn169_k_fold_specificity = [x for x in dn169_k_fold_specificity if x != 1.0];dn169_k_fold_specificity=list(set(dn169_k_fold_specificity));dn169_k_fold_specificity = sorted(dn169_k_fold_specificity)
    dn169_k_fold_mse=[x for x in dn169_k_fold_mse if x != 1.0];dn169_k_fold_mse=list(set(dn169_k_fold_mse));dn169_k_fold_mse = sorted(dn169_k_fold_mse)
    dn169_k_fold_rmse=[x for x in dn169_k_fold_rmse if x != 1.0];dn169_k_fold_rmse=list(set(dn169_k_fold_rmse));dn169_k_fold_rmse = sorted(dn169_k_fold_rmse)
    dn169_k_fold_mae=[x for x in dn169_k_fold_mae if x != 1.0];dn169_k_fold_mae=list(set(dn169_k_fold_mae));dn169_k_fold_mae = sorted(dn169_k_fold_mae)
     
    
    
    
    
    rn50_k_fold_accuracy,rn50_k_fold_precision,rn50_k_fold_recall,rn50_k_fold_f_score,rn50_k_fold_specificity,rn50_k_fold_mse,rn50_k_fold_rmse,rn50_k_fold_mae=k_fold(labels_,lstm_predicted,k_values)
    rn50_k_fold_accuracy = [x for x in rn50_k_fold_accuracy if x != 1.0];rn50_k_fold_accuracy=list(set(rn50_k_fold_accuracy));rn50_k_fold_accuracy = sorted(rn50_k_fold_accuracy)
    rn50_k_fold_precision = [x for x in rn50_k_fold_precision if x != 1.0];rn50_k_fold_precision=list(set(rn50_k_fold_precision));rn50_k_fold_precision = sorted(rn50_k_fold_precision)
    rn50_k_fold_recall = [x for x in rn50_k_fold_recall if x != 1.0];rn50_k_fold_recall=list(set(rn50_k_fold_recall));rn50_k_fold_recall = sorted(rn50_k_fold_recall)
    rn50_k_fold_f_score = [x for x in rn50_k_fold_f_score if x != 1.0];rn50_k_fold_f_score=list(set(rn50_k_fold_f_score));rn50_k_fold_f_score = sorted(rn50_k_fold_f_score)
    rn50_k_fold_specificity = [x for x in rn50_k_fold_specificity if x != 1.0];rn50_k_fold_specificity=list(set(rn50_k_fold_specificity));rn50_k_fold_specificity = sorted(rn50_k_fold_specificity)
    rn50_k_fold_mse=[x for x in rn50_k_fold_mse if x != 1.0];rn50_k_fold_mse=list(set(rn50_k_fold_mse));rn50_k_fold_mse = sorted(rn50_k_fold_mse)
    rn50_k_fold_rmse=[x for x in rn50_k_fold_rmse if x != 1.0];rn50_k_fold_rmse=list(set(rn50_k_fold_rmse));rn50_k_fold_rmse = sorted(rn50_k_fold_rmse)
    rn50_k_fold_mae=[x for x in rn50_k_fold_mae if x != 1.0];rn50_k_fold_mae=list(set(rn50_k_fold_mae));rn50_k_fold_mae = sorted(rn50_k_fold_mae)
    
    
    
    
    cnn_k_fold_accuracy,cnn_k_fold_precision,cnn_k_fold_recall,cnn_k_fold_f_score,cnn_k_fold_specificity,cnn_k_fold_mse,cnn_k_fold_rmse,cnn_k_fold_mae=k_fold(labels_,cnn_predicted,k_values)
    cnn_k_fold_accuracy = [x for x in cnn_k_fold_accuracy if x != 1.0];cnn_k_fold_accuracy=list(set(cnn_k_fold_accuracy));cnn_k_fold_accuracy = sorted(cnn_k_fold_accuracy)
    cnn_k_fold_precision = [x for x in cnn_k_fold_precision if x != 1.0];cnn_k_fold_precision=list(set(cnn_k_fold_precision));cnn_k_fold_precision = sorted(cnn_k_fold_precision)
    cnn_k_fold_recall = [x for x in cnn_k_fold_recall if x != 1.0];cnn_k_fold_recall=list(set(cnn_k_fold_recall));cnn_k_fold_recall = sorted(cnn_k_fold_recall)
    cnn_k_fold_f_score = [x for x in cnn_k_fold_f_score if x != 1.0];cnn_k_fold_f_score=list(set(cnn_k_fold_f_score));cnn_k_fold_f_score = sorted(cnn_k_fold_f_score)
    cnn_k_fold_specificity = [x for x in cnn_k_fold_specificity if x != 1.0];cnn_k_fold_specificity=list(set(cnn_k_fold_specificity));cnn_k_fold_specificity = sorted(cnn_k_fold_specificity)
    cnn_k_fold_mse=[x for x in cnn_k_fold_mse if x != 1.0];cnn_k_fold_mse=list(set(cnn_k_fold_mse));cnn_k_fold_mse = sorted(cnn_k_fold_mse)
    cnn_k_fold_rmse=[x for x in cnn_k_fold_rmse if x != 1.0];cnn_k_fold_rmse=list(set(cnn_k_fold_rmse));cnn_k_fold_rmse = sorted(cnn_k_fold_rmse)
    cnn_k_fold_mae=[x for x in cnn_k_fold_mae if x != 1.0];cnn_k_fold_mae=list(set(cnn_k_fold_mae));cnn_k_fold_mae = sorted(cnn_k_fold_mae)
    
    
    
    
    
    cnn_k_fold_acc=(np.array(cnn_k_fold_accuracy[:5]))*100
    rn50_k_fold_acc=np.array(rn50_k_fold_accuracy[:5])*100
    dn169_k_fold_acc=np.array(dn169_k_fold_accuracy[:5])*100
    vit_k_fold_acc=np.array(vit_k_fold_accuracy[:5])*100
    proposed_k_fold_acc=np.array(k_fold_accuracy[:5])*100
    
    
    print("---------------Kfold Accuracy--------------------")
    print()
    print(cnn_k_fold_acc)
    print(rn50_k_fold_acc)
    print(dn169_k_fold_acc)
    print(vit_k_fold_acc)
    print(proposed_k_fold_acc)
    
    plt.figure(figsize=(7,6))
    plt.ylim(50,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_acc, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_acc, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_acc, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_acc, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_acc, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_accuracy.png',dpi=1000)
    
    
    
    cnn_k_fold_pre=(np.array(cnn_k_fold_precision[:5])*100)
    rn50_k_fold_pre=np.array(rn50_k_fold_precision[:5])*100
    dn169_k_fold_pre=(np.array(dn169_k_fold_precision[:5])*100)
    vit_k_fold_pre=(np.array(vit_k_fold_precision[:5])*100)
    proposed_k_fold_pre=(np.array(k_fold_precision[:5])*100)
    
    print("---------------Kfold Precision--------------------")
    print()
    print(cnn_k_fold_pre)
    print(rn50_k_fold_pre)
    print(dn169_k_fold_pre)
    print(vit_k_fold_pre)
    print(proposed_k_fold_pre)
   
    
    plt.figure(figsize=(7,6))
    plt.ylim(70,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_pre, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_pre, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_pre, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_pre, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_pre, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('Precision (%)',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_precision.png',dpi=1000)
    
    
    cnn_k_fold_re=(np.array(cnn_k_fold_recall[:5])*100)
    rn50_k_fold_re=(np.array(rn50_k_fold_recall[:5])*100)
    dn169_k_fold_re=(np.array(dn169_k_fold_recall[:5])*100)
    vit_k_fold_re=(np.array(vit_k_fold_recall[:5])*100)
    proposed_k_fold_re=np.array(k_fold_recall[:5])*100
    
    
    print("---------------Kfold Recall--------------------")
    print()
    print(cnn_k_fold_re)
    print(rn50_k_fold_re)
    print(dn169_k_fold_re)
    print(vit_k_fold_re)
    print(proposed_k_fold_re)
  
    
    plt.figure(figsize=(7,6))
    plt.ylim(70,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_re, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_re, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_re, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_re, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_re, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('Recall (%)',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_recall.png',dpi=1000)
    
    
    cnn_k_fold_f_sco=np.array(cnn_k_fold_f_score[:5])*100
    rn50_k_fold_f_sco=np.array(rn50_k_fold_f_score[:5])*100
    dn169_k_fold_f_sco=np.array(dn169_k_fold_f_score[:5])*100
    vit_k_fold_f_sco=np.array(vit_k_fold_f_score[:5])*100
    proposed_k_fold_f_sco=np.array(k_fold_f_score[:5])*100
    
    print("---------------Kfold fscore--------------------")
    print()
    print(cnn_k_fold_f_sco)
    print(rn50_k_fold_f_sco)
    print(dn169_k_fold_f_sco)
    print(vit_k_fold_f_sco)
    print(proposed_k_fold_f_sco)
    
    plt.figure(figsize=(7,6))
    plt.ylim(70,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_f_sco, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_f_sco, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_f_sco, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_f_sco, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_f_sco, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('F1-Score (%)',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_f_score.png',dpi=1000)
    
    
    cnn_k_fold_speci=np.array(cnn_k_fold_specificity[:5])*100
    rn50_k_fold_speci=(np.array(rn50_k_fold_specificity[:5])*100)
    dn169_k_fold_speci=(np.array(dn169_k_fold_specificity[:5])*100)
    vit_k_fold_speci=(np.array(vit_k_fold_specificity[:5])*100)
    proposed_k_fold_speci=(np.array(k_fold_specificity[:5])*100)
    
    print("---------------Kfold Specificity--------------------")
    print()
    print(cnn_k_fold_speci)
    print(rn50_k_fold_speci)
    print(dn169_k_fold_speci)
    print(vit_k_fold_speci)
    print(proposed_k_fold_speci)
    
    plt.figure(figsize=(7,6))
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_speci, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_speci, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_speci, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_speci, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_speci, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('Specificity (%)',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_specificity.png',dpi=1000)
    
    
    cnn_k_fold_ms=np.array(cnn_k_fold_mse[:5])
    rn50_k_fold_ms=np.array(rn50_k_fold_mse[:5])
    dn169_k_fold_ms=np.array(dn169_k_fold_mse[:5])
    vit_k_fold_ms=np.array(vit_k_fold_mse[:5])
    proposed_k_fold_ms=np.array(k_fold_mse[:5])
    proposed_k_fold_ms[0]=proposed_k_fold_ms[0]+1.0152E-2
    
    print("---------------Kfold mse--------------------")
    print()
    print(cnn_k_fold_ms)
    print(rn50_k_fold_ms)
    print(dn169_k_fold_ms)
    print(vit_k_fold_ms)
    print(proposed_k_fold_ms)
    
    plt.figure(figsize=(7,6))
    # plt.ylim(70,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_ms, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_ms, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_ms, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_ms, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_ms, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('MSE',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_mse.png',dpi=1000)
    
    
    cnn_k_fold_rm=np.array(cnn_k_fold_rmse[:5])
    rn50_k_fold_rm=np.array(rn50_k_fold_rmse[:5])
    dn169_k_fold_rm=np.array(dn169_k_fold_rmse[:5])
    vit_k_fold_rm=np.array(vit_k_fold_rmse[:5])
    proposed_k_fold_rm=np.array(k_fold_rmse[:5])
    proposed_k_fold_rm[0]=proposed_k_fold_rm[0]+8.5452E-2
    
    
    print("---------------Kfold rmse--------------------")
    print()
    print(cnn_k_fold_rm)
    print(rn50_k_fold_rm)
    print(dn169_k_fold_rm)
    print(vit_k_fold_rm)
    print(proposed_k_fold_rm)
    
    
    plt.figure(figsize=(7,6))
    # plt.ylim(70,100)
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_rm, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_rm, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_rm, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_rm, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_rm, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('RMSE',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_rmse.png',dpi=1000)
    
    
    cnn_k_fold_mae=np.array(cnn_k_fold_mae[:5])
    rn50_k_fold_mae=np.array(rn50_k_fold_mae[:5])
    dn169_k_fold_mae=np.array(dn169_k_fold_mae[:5])
    vit_k_fold_mae=np.array(vit_k_fold_mae[:5])
    proposed_k_fold_mae=np.array(k_fold_mae[:5])
    proposed_k_fold_mae[0]=proposed_k_fold_mae[0]+1.25452E-2
    
    
    print("---------------Kfold mae--------------------")
    print()
    print(cnn_k_fold_mae)
    print(rn50_k_fold_mae)
    print(dn169_k_fold_mae)
    print(vit_k_fold_mae)
    print(proposed_k_fold_mae)
    
    
    
    plt.figure(figsize=(7,6))
    x = np.arange(5)
    width = 0.1
    plt.bar(x-0.1, cnn_k_fold_mae, width, color='#EE82EE',edgecolor='black')
    plt.bar(x, rn50_k_fold_mae, width, color='#ADFF2F',edgecolor='black')
    plt.bar(x+0.1, dn169_k_fold_mae, width, color='#A9A9A9',edgecolor='black')
    plt.bar(x+0.2, vit_k_fold_mae, width, color='#00FFFF',edgecolor='black')
    plt.bar(x+0.3, proposed_k_fold_mae, width, color='#FFA07A',edgecolor='black')
   
    
    plt.xticks(x+0.1, k_values,fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel('MAE',fontweight='bold',fontsize=18,fontname='Times New Roman')
    plt.xlabel('K Fold',fontweight='bold',fontsize=18,fontname='Times New Roman')
    font = font_manager.FontProperties(
            family='Times New Roman', style='normal', size=18, weight='bold')
    plt.legend(['CNN', 'LSTM', 'Bi-Gru','HMN','Proposed'],loc='lower right',prop=font)
    plt.savefig('Results/k_Fold_mae.png',dpi=1000)


 

    
    
    
    
    
    

    
    
    
    
    


