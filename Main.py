import numpy as np 
import pandas as pd
import cv2
from Preprocess import *
from Segmentation import *
from Existing_Seg import *
from Optimization import *
from Existing_Class import *
from Classification import *
from keras.models import load_model


def Train():
    "Reading Images "
    images=[]
    masks=[]
    preprocesed_images=[]
    
    df = pd.read_csv('OTU_2d-20241021T080321Z-001/OTU_2d/train_cls.txt', delim_whitespace=True)
    image_names=df.iloc[:,0]
    labels=df.iloc[:,1]
    
    
    # Preprocess
    # Anisotropic Speckle Reducing Diffusion filter
    for image_n in image_names:
        image=cv2.imread("OTU_2d-20241021T080321Z-001/OTU_2d/images/"+image_n)
        image_=cv2.resize(image,(256,256))
        filtered_image = anisotropic_speckle_reducing_filter(image_)
        normalized_image=filtered_image/255.0
        # print(normalized_image.shape)
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
        
    # np.save("Features/Original_images.npy",images)
    # np.save("Features/masks.npy",masks)
    # np.save("Features/preprocesed_images.npy",preprocesed_images)
    # np.save("Features/labels.npy",labels)
    
    
    
    # segmentation
    # Progressive attention attached Dual embedding layer Vision transformer (PA-DViT)
    pre_images = np.load("Features/preprocesed_images.npy")
    masks_images = np.load("Features/masks.npy")
    
    model = PA_DViT_model(pre_images)
    model.fit(pre_images, masks_images, batch_size=32, epochs=100, validation_split=0.2)
    seg_model=load_model("Segmentation Model/proposed_seg_model.h5")
    predicted_images=seg_model.predict(pre_images)
    thresholded_images = [np.where(pred_image > 0.5, 1, 0) for pred_image in predicted_images]
    thresholded_images=np.array(thresholded_images)
    thresholded_images=thresholded_images.reshape(thresholded_images.shape[0],thresholded_images.shape[1],thresholded_images.shape[2])
    
    segmented_images=[]
    for pred_image,pre_image in zip(thresholded_images,pre_images):
        img = pred_image.reshape(256,256,1)
        org=(pre_image*img)*255
        org = org.astype("uint8")
        segmented_images.append(org)
    # np.save("Features/segmented_images.npy",segmented_images)
    # np.save("Features/predicted_masks.npy",thresholded_images)
        
    #Existing
    
    pyramid_vit_model = pyramid_vit(pim)
    pyramid_vit_model.fit(pim, mim, epochs=100, batch_size=32,validation_split=0.2)
    
    # Residual Vision Transformaer
    residual_vit_model = residual_vit_segmentation_model(pre_images)
    residual_vit_model.fit(pre_images, masks_images, epochs=100, batch_size=32,validation_split=0.2)
    
    # Deep Aggregation Vision Transformer 
    deep_aggregation_vit_model = deep_aggregation_vit_block(pre_images)
    deep_aggregation_vit_model.fit(pre_images, masks_images, epochs=100, batch_size=32,validation_split=0.2)
    
    
    # hierarchical Vision Transformer 
    hierarchical_model = hierarchical_vit(pre_images)
    hierarchical_model.fit(pre_images, masks_images, epochs=100, batch_size=32,validation_split=0.2)
    
    
    # classification
    # Detection
    # Attention based Convolutional Hidden Markow Network (ACHMN)
    # Integrated Dung goose optimization algorithm
    labels=np.load("Features/labels.npy")
    segmented_images=np.arrray(segmented_images)/255.0
    
    num_classes = len(np.unique(labels))
    model = CNNHMC(num_classes=num_classes)
    model.build(input_shape=(None, 256, 256, 3))  
    opt=HybridGGO_DBO()
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Fit the model with dummy data
    model.fit(segmented_images, labels,batch_size=32, epochs=100,validation_split=0.2)
    
    
    
    #Existing
    
    #CNN
    cnn_model=cnn_model(segmented_images,labels)
    cnn_model.fit(segmented_images,labels,batch_size=32,epochs=100,validation_split=0.2)
    
    
    
    # LSTM
    lstm_model_=lstm_model(segmented_images,labels)
    lstm_model_.fit(segmented_images,labels,batch_size=32,epochs=100,validation_split=0.2)
    
    # BIGRU
    bigru_model_=bigru_model(segmented_images,labels)
    bigru_model_.fit(segmented_images,labels,batch_size=32,epochs=100,validation_split=0.2)
    
    # Hierarchical Memory Network
    hmn_model_=hmn_model(segmented_images,labels)
    hmn_model_.fit(segmented_images,labels,batch_size=32,epochs=100,validation_split=0.2)
    
    
    
    
    
    
