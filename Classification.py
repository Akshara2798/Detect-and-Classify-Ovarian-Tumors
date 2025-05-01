import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from Optimization import *

class CNNHMC(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNNHMC, self).__init__()

        # CNN layers
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.pool = layers.MaxPooling2D((2, 2))
        self.dropout = layers.Dropout(0.25)

        # Flatten layer to prepare for fully connected layers
        self.flatten = layers.Flatten()

        # Hierarchical classification heads (HMC blocks)
        self.hmc1 = layers.Dense(num_classes, activation='softmax')
        self.hmc2 = layers.Dense(num_classes, activation='softmax')
        self.hmcN = layers.Dense(num_classes, activation='softmax')

        # Final concatenation layer
        self.fc_concat = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    # The call method should be defined outside of the __init__ method
    def call(self, inputs):
        # CNN feature extraction
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
    
        # Flatten the feature map for fully connected layers
        x = self.flatten(x)
    
        # Hierarchical classification outputs
        hmc1_out = self.hmc1(x)
        hmc2_out = self.hmc2(x)
        hmcN_out = self.hmcN(x)
    
        # Concatenate the hierarchical outputs
        concat_out = tf.concat([hmc1_out, hmc2_out, hmcN_out], axis=1)
    
        # Fully connected layers after concatenation
        fc_out = self.fc_concat(concat_out)
        final_out = self.output_layer(fc_out)
    
        return final_out  # Return only the final output


