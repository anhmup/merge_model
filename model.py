# %%
import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate , LeakyReLU, BatchNormalization , ReLU , Embedding , LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# from keras.engine import Layer, InputSpec
# from tensorflow.keras.engine.topology import Layer
import keras.backend as K
import pandas as pd
import numpy as np
import ast
import cv2
# import imutils
import matplotlib.pyplot as plt
from PIL import Image

# %% [markdown]
# # Model

# %%
def convert_String_to_array(string_array):
    nested_list = ast.literal_eval(string_array)

    return nested_list

# %%
def identity_block(input_tensor, kernel_size, filter_num, stage):
    x = layers.Conv2D(filter_num, (1,1))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter_num, kernel_size, padding ='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x 

# %%
def conv_block(input_tensor,kernel_size,filter_num,stage,strides=(2, 2)):
    x = layers.Conv2D(filter_num, (1, 1), strides=strides)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter_num, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2D(filter_num, (1, 1), strides=strides)(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x 

# %%
def res_net_18(input_tensor):
    #input_tensor = Input(shape=(height, width, depth))

    x = layers.Conv2D(64, (7, 7),strides=(2, 2),padding='same',name='input')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding = 'same')(x)

    x = conv_block(x, 3, 64, stage=2, strides=(1, 1))
    c2 = identity_block(x, 3, 64, stage=2)

    x = conv_block(c2, 3, 128, stage=3)
    c3 = identity_block(x, 3, 128, stage=3)

    x = conv_block(c3, 3, 256, stage=4)
    c4 = identity_block(x, 3, 256, stage=4)

    x = conv_block(c4, 3, 512, stage=5)
    c5 = identity_block(x, 3, 512, stage=5)

    #model = Model(inputs=input_tensor , outputs = x)
    return c2, c3 , c4 ,c5

# %%
def lateral(x, out_channels):
    x = layers.Conv2D(out_channels, (1, 1), padding='same')(x)
    return  x
def upsampling_add(x,y):
    b,h,w,c = y.shape
    x = tf.image.resize(x, (h,w), method='bilinear')
    return layers.add([x,y])
def smooth(x):
    x = Conv2D(256, kernel_size = 3, strides = 1, padding ='same')(x)
    return x

# %%
def transformer_encoder(inputs, num_heads, mlp_dim, dropout_rate):
    # Layer normalization 1
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Multi-head attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=dropout_rate)(x, x)
    # Skip connection 1
    x = layers.Add()([x, inputs])

    # Layer normalization 2
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    # MLP (feed-forward network)
    y = layers.Dense(mlp_dim, activation=tf.nn.gelu)(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    # Skip connection 2
    return layers.Add()([y, x])

# %%
data_col = pd.read_csv('data_col.csv')
data_row = pd.read_csv('data_row.csv')


# %%
class Roi_Pooling(layers.Layer):
    def __init__(self, pool_size, **kwargs):
        super(Roi_Pooling, self).__init__(**kwargs)
        self.pool_size = pool_size
    def call(self, feature_map, rois):
        """
        feature _map  = [b,h,w,c]
        rois = [x_loc, y_loc , x1 ,y1, x2, y2]

        """
        b,h,w,c = feature_map.get_shape().as_list()
        outputs = []
        x1 = rois[:,2]/w
        y1 = rois[:,3]/h
        x2 = rois[:,4]/w
        y2 = rois[:,5]/h
        boxes = tf.stack([y1, x1, y2, x2], axis=1)
        batch_indices = tf.zeros(shape=(tf.shape(rois)[0]), dtype=tf.int32)
        
        pooled_feature = tf.image.crop_and_resize(feature_map, boxes, box_indices=batch_indices, crop_size=[self.pool_size[0], self.pool_size[1]], method="bilinear")
        return pooled_feature

    def get_config(self):
        config = super().get_config()
        config.update({'pool_size' : self.pool_size})
        return config

# %%
MAX_ = 30

# %%
class PatchEncoder(layers.Layer):
    def __init__(self , num_patches = (MAX_*MAX_), projection_dims = 512,**kwargs ):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.num_dims  = projection_dims
        self.x_positional_embeddings = Embedding(input_dim = num_patches, output_dim = projection_dims)
        self.y_positional_embeddings = Embedding(input_dim = num_patches, output_dim = projection_dims)

    def call(self, x , rois):
        """
        x: input tensor after roi pooling ( shape = ((M x N), 512))
        position embedding: (shape = (50, 512))
        """
        # x = tf.expand_dims(x, axis=0)
        print(x.shape)
        pos_list = []
        x_position = tf.range(0,limit=self.num_patches, delta=1)
        y_position = tf.range(0,limit=self.num_patches, delta=1)
        ngang = tf.reduce_max(rois[:,1])
        doc  = tf.reduce_max(rois[:,0])
        num_ = tf.shape(x)[0]
        pad_len = (MAX_*MAX_) - tf.shape(x)[0]
        x_padded = tf.pad(x, [[0, pad_len],[0,0]], constant_values=0)
        print("here")
        print(x_padded.shape)
        # pos_ = tf.zeros(shape=tf.shape(x_padded))
        for i in range(MAX_*MAX_):
            x_idx = tf.math.floormod(i, MAX_)
            y_idx = tf.math.floordiv(i, MAX_)
            x_pos = self.x_positional_embeddings(x_idx) # (512,)
            y_pos = self.y_positional_embeddings(y_idx) # (512,)
            pos_embedded = x_pos + y_pos
            pos_list.append(pos_embedded)
        pos_ = tf.stack(pos_list,axis=0)

        print(f'pos_ {pos_.shape}')
        encoded_full = x_padded + pos_
        return encoded_full

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_paches": self.num_patches,
            "d": self.num_dims,
        })
        return config


# %%
def mlp(x):
    x = Dense(3072, activation="gelu")(x)
    x = Dropout(0.1)(x)
    x = Dense(512)(x)
    x = Dropout(0.1)(x)
    return x
def transformer_encoder(x):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=10, key_dim=512
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x)
    x = Add()([x, skip_2])

    return x

# %%
INPUT_SHAPE =(256,256,3)
input_tensor = Input(shape=(INPUT_SHAPE), batch_size= 1 , name= "image" )
input_cell_box = Input(shape=(None,6), batch_size= 1 , name="bbox")  
c2, c3 , c4 ,c5  = res_net_18(input_tensor)

m5 = lateral(c5, 256)

m4  = upsampling_add(m5,lateral(c4,256) )
m3  = upsampling_add(m4,lateral(c3,256) )
m2  = upsampling_add(m3,lateral(c2,256) )

p5 = smooth(m5)
p4 = smooth(m4)
p3 = smooth(m3)
p2 = smooth(m2)

p2_2 = Conv2D(128, kernel_size = 3, strides = 1, padding ='same', name = "p2_2")(p2)


crops = Roi_Pooling(pool_size=(7,7))(p2_2, input_cell_box[0])

# #position embedded ing
embedded_patches= tf.keras.layers.Flatten()(crops)
embedded_patches= layers.Dense(512)(embedded_patches)
embedded_patches = layers.Activation('ReLU')(embedded_patches)
embedded_patches= layers.Dense(512)(embedded_patches)

encoded = PatchEncoder()(embedded_patches , input_cell_box) # 50*50 patches, each patch has 512 dims

encoded = tf.expand_dims(encoded, axis = 0)

cls_token = tf.zeros((1, 1, 512))

x = tf.concat([cls_token, encoded], axis = 1)

for _ in range(10):
    x = transformer_encoder(x)
x = LayerNormalization()(x) 
x = x[:, 0, :] 
x = Dropout(0.1)(x)
x_sub = Dense((MAX_*MAX_), activation="softmax")(x)

row = Dense((MAX_*MAX_), activation="softmax" , name = "row")(x_sub)
col = Dense((MAX_*MAX_), activation="softmax" , name = "col")(x_sub)

input_ = [input_tensor, input_cell_box]
out = [row, col]

model = Model(inputs = input_, outputs = out)

# %%
embedded_patches.shape

# %%
test_p22 = tf.random.uniform((1,64,64,128), minval=0, maxval=256, dtype= tf.float32, seed= None, name=None)
test_bbox = tf.constant([[0,0,5,5, 10, 20], [0,1,10,15,14,17]] , dtype= tf.float32)
a = Roi_Pooling(pool_size=(7,7))(test_p22, test_bbox)
b = tf.keras.layers.Flatten()(a)
b.shape

# %%
np.all(b[0] ==0) 

# %%
crops.shape

# %%
model.compile(loss= {'row' : 'binary_crossentropy', 'col' : 'binary_crossentropy'}, optimizer='adam')

# %%
model.output_shape

# %%
model.summary()

# %%
model.input_shape

# %%
random_input_1 = tf.random.normal(shape=(1,256,256,3))
random_input_2 =tf.constant([[[0,0,10,20,30,40], [0,1,50,60,70,80],[1,0,5,5, 10, 20], [1,1,10,15,14,17]]])
print(random_input_2.shape)
row, col = model.predict([random_input_1, random_input_2])
print(row.shape)
print(col.shape)

# %%
row

# %% [markdown]
# # Data loader 

# %%
data_path =       'D:/learn/de cuong/code/split/data_generated/image/'
source_xml_path = 'D:/learn/de cuong/code/split/data_pubtables_1M/PubTables-1M-Structure_Annotations_Test/'
source_word_path ='D:/learn/de cuong/code/split/data_pubtables_1M/PubTables-1M-Structure_Table_Words/'


saved_txt_row_path ="D:/learn/de cuong/code/merge/data/row/"
saved_txt_col_path ="D:/learn/de cuong/code/merge/data/col/"

# %%
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import requests
import tarfile

from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt

from matplotlib.patches import  Rectangle
from PIL import ImageFont, ImageDraw
import tensorflow as tf

# %%
def convert_String_to_array(string_array)
    nested_list = ast.literal_eval(string_array)

    return nested_list

# %%
def normalize_img(img):
    #norm_img = (img - img.min()) / (img.max() - img.min())
    norm_img = np.array(img,dtype= np.float32)/255.0
    return norm_img

# %%
# input image = (256,256)  ---> output of FPN is (64,64)

# %%
def get_cell_list(row, col):
    cell_list = []
    cord_list = []
    for idx_row, box_row in enumerate(row):
        x1_row, y1_row , x2_row, y2_row = box_row
        for idx_col, box_col in enumerate(col):
            x1_col, y1_col, x2_col, y2_col = box_col

            update_cell = [x1_col, y1_row, x2_col, y2_row]
            update_cord = [idx_row, idx_col]
            cell_list.append(update_cell)
            cord_list.append(update_cord)
    current_len = len(cell_list)
    cell_list = np.array(cell_list)
    pad = (0, 2500-current_len)
    cell_list_padded = np.pad(cell_list, (pad, (0,0)), mode= 'constant', constant_values= 0)
    return cell_list_padded, cord_list

# %%
def box_feature_map(cell_list):
    new_cell_list = []
    for box in cell_list:
        x1 , y1 , x2 , y2 = box
        x1 = int((x1*64)/256)
        y1 = int((y1*64)/256)
        x2 = int((x2*64)/256)
        y2 = int((y2*64)/256)
        new_box = [x1, y1, x2, y2]
        new_cell_list.append(new_box)
    new_cell_list = np.array(new_cell_list)
    return new_cell_list

# %%
def load_data_input(file_name, col_bbox_list, row_bbox_list):
    col = convert_String_to_array(col_bbox_list)
    col =np.array(col)
    col = col[0]

    row= convert_String_to_array(row_bbox_list)
    row =np.array(row)
    row = row[0]


    img_path  = data_path + file_name + '.jpg'
    image = Image.open(img_path)
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (256,256))
    image_np = np.array(image_np)

    cell_list, cord_list = get_cell_list(row, col) # cell_list len 2500
    feature_map_cell = box_feature_map(cell_list) # cell_list len 2500


    return image_np, feature_map_cell

# %%
test_file_name = "PMC1064078_table_0"
row_arr = np.load(saved_txt_row_path + test_file_name + ".npy")
row_arr
pad_h = (0,(50- row_arr.shape[0]))
pad_w = (0, (50- row_arr.shape[1]))

padded = np.pad(row_arr, (pad_h, pad_w), mode= 'constant', constant_values= 0)


# %%
y = np.reshape(padded, (2500))
y.shape

# %%
model.output_shape

# %%
def pad_array(x):
    pad_h = (0,(50- x.shape[0]))
    pad_w = (0, (50- x.shape[1]))
    padded = np.pad(x, (pad_h, pad_w), mode= 'constant', constant_values= 0)
    return padded
def load_data_gt(file_name):
    row_arr = np.load(saved_txt_row_path + file_name + ".npy")
    col_arr = np.load(saved_txt_col_path + file_name + ".npy")
    
    row_arr_padded = pad_array(row_arr)
    col_arr_padded = pad_array(col_arr)
    row_arr_padded = np.reshape(row_arr_padded, (2500))
    col_arr_padded = np.reshape(col_arr_padded, (2500))
    return row_arr_padded, col_arr_padded

# %%
class DataGenerator(tf.keras.utils.Sequence) :
    def __init__(self, data_row, data_col , batch_size= 1,dim = 256, shuffle=True ):
        self.data_row= data_row
        self.data_col = data_col
        self.indices= self.data_row.index.tolist()
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.data_row) / self.batch_size))
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        image_input_batch, feature_map_bbox_batch, gt_row_batch, gt_col_batch = self.data_generation(batch)
        return {"image" : image_input_batch, "bbox" : feature_map_bbox_batch} , {"row" :  gt_row_batch, "col": gt_col_batch}

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def data_generation(self, batch):
        image_input_batch = np.empty((self.batch_size, self.dim , self.dim , 3))
        feature_map_bbox_batch = np.empty((self.batch_size, 2500, 4))
        gt_row_batch = np.empty((self.batch_size, 2500))
        gt_col_batch = np.empty((self.batch_size, 2500))

        file_name = self.data_col.iloc[0][0]
        col_bbox_list = self.data_col.iloc[0][1]
        row_bbox_list = self.data_row.iloc[0][1]

        image_resized , feature_map_bbox = load_data_input(file_name, col_bbox_list, row_bbox_list)
        image_resized = normalize_img(image_resized)
        row_arr_padded, col_arr_padded = load_data_gt(file_name)

        image_input_batch[0] = image_resized
        feature_map_bbox_batch[0] = feature_map_bbox

        gt_row_batch[0] = row_arr_padded
        gt_col_batch[0] = col_arr_padded


        return image_input_batch, feature_map_bbox_batch, gt_row_batch, gt_col_batch

# %%
data_col = pd.read_csv('data_col.csv')
data_row = pd.read_csv('data_row.csv')

# %%
i = 0
input_, gt  = next(iter(DataGenerator(data_col=data_col[i:], data_row=data_row[i:])))

# %% [markdown]
# # test training 

# %%
training_generator = DataGenerator(data_col=data_col[:10].reset_index(drop=True), data_row=data_row[:10].reset_index(drop=True))

# %%
# history = model.fit(training_generator, epochs= 1)


