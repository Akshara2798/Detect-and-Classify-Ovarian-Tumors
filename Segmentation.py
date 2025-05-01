import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Dual Embedding Layer
class DualEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches, embed_dim):
        super(DualEmbeddingLayer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        self.spatial_embedding = tf.keras.layers.Dense(embed_dim)
        self.pos_embedding = self.add_weight(
            shape=(1, num_patches, embed_dim), initializer='random_normal', trainable=True
        )
    
    def call(self, x):
        patches = tf.image.extract_patches(images=x, 
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1], padding='VALID')
        
        patches = tf.reshape(patches, (tf.shape(patches)[0], -1, self.embed_dim))
        spatial_embeddings = self.spatial_embedding(patches)
        embeddings = spatial_embeddings + self.pos_embedding
        return embeddings

# Progressive Attention Layer
class ProgressiveAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(ProgressiveAttention, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        attention_out = self.multi_head_attention(x, x)
        x = self.norm(x + attention_out)
        return x

# Vision Transformer Block
class VisionTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(VisionTransformerBlock, self).__init__()
        self.encoder_layers = [ProgressiveAttention(embed_dim, num_heads) for _ in range(num_layers)]
    
    def call(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


# Segmentation Decoder
class SegmentationDecoder(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super(SegmentationDecoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same') # Final layer to match 256x256
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# PA-DViT Model for Segmentation
def PA_DViT_model(images):
    
    input_shape = images[0].shape
    patch_size = 16              # Patch size for Vision Transformer
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    embed_dim = 768              # Embedding dimension
    num_layers = 12              # Number of transformer layers
    num_heads = 12               # Number of attention heads
    output_channels = 1          # Segmentation output (binary)
    inputs = tf.keras.layers.Input(shape=input_shape)
    embeddings = DualEmbeddingLayer(patch_size, num_patches, embed_dim)(inputs)
    transformer_out = VisionTransformerBlock(num_layers, embed_dim, num_heads)(embeddings)
    transformer_out = tf.reshape(transformer_out, [-1, input_shape[0] // patch_size, input_shape[1] // patch_size, embed_dim])
    segmentation_map = SegmentationDecoder(output_channels)(transformer_out)
    
    model = tf.keras.Model(inputs, segmentation_map)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



