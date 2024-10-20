# %%
import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate , LeakyReLU, BatchNormalization , ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K
import pandas as pd
import numpy as np
import cv2
# import imutils
import matplotlib.pyplot as plt
from PIL import Image

# %% [markdown]
# # load data 

# %%


# %% [markdown]
# # model

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

input_tensor = Input(shape=(512, 512, 3))
c2, c3 , c4 ,c5  = res_net_18(input_tensor)
# print(c2.shape)
m5 = lateral(c5, 256)

m4  = upsampling_add(m5,lateral(c4,256) )
m3  = upsampling_add(m4,lateral(c3,256) )
m2  = upsampling_add(m3,lateral(c2,256) )

p5 = smooth(m5)
p4 = smooth(m4)
p3 = smooth(m3)
p2 = smooth(m2)

p2_2 = Conv2D(16, kernel_size = 3, strides = 1, padding ='same')(p2)

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
mlp_dim = 512
INPUT_SHAPE = (256,256,3)

# %%
data_col = pd.read_csv('data_col.csv')
data_row = pd.read_csv('data_row.csv')

# %%
import ast
def convert_String_to_array(string_array):
    nested_list = ast.literal_eval(string_array)

    return nested_list

# %%
file_name = data_col.iloc[0][0]
col_bbox_list = data_col.iloc[0][1]
row_bbox_list = data_row.iloc[0][1]
file_name


# %%
col = convert_String_to_array(col_bbox_list)
col =np.array(col)
col = col[0]

row= convert_String_to_array(row_bbox_list)
row =np.array(row)
row = row[0]

# %%
row

# %%
col

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

    return cell_list, cord_list


# %%
data_path =       'D:/learn/de cuong/code/split/data_generated/image/'
source_xml_path = 'D:/learn/de cuong/code/split/data_pubtables_1M/PubTables-1M-Structure_Annotations_Test/'
source_word_path ='D:/learn/de cuong/code/split/data_pubtables_1M/PubTables-1M-Structure_Table_Words/'


saved_txt_row_path ="D:/learn/de cuong/code/merge/data_merge_cell/row/"
saved_txt_col_path ="D:/learn/de cuong/code/merge/data_merge_cell/col/"

# %%
def show_test_box(img_np, table_spaning_cell):
    for box in table_spaning_cell:
        x1 , y1 , x2 , y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_np

# %%
img_path  = data_path + file_name + '.jpg'
image = Image.open(img_path)
image_np = np.array(image)
image_np = cv2.resize(image_np, (256,256))
image_np = np.array(image_np)
img_np = show_spaning_cell(image_np, cell_list)
plt.imshow(img_np)
plt.show()

# %%
cell_list, cord_list = get_cell_list(row, col)

# %%
def pre_load_data(file_name, row, col):
    img_path  = data_path + file_name + '.jpg'
    image = Image.open(img_path)
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (256,256))
    image_np = np.array(image_np)

    cell_list, cord_list = get_cell_list(row, col)

    return image_np, cell_list, cord_list

def extract_patches(image, cell_list, cord_list):
    patches_list = []
    for box in cell_list:
        x1, y1, x2, y2 = box
        patch = image[y1:y2, x1:x2, :]
        patches_list.append(patch)
    return patches_list
    

# %%
image , cell_list, cord_list = pre_load_data(file_name, row, col)
patches_list  = extract_patches(image, cell_list, cord_list)

# %%
def show_patch_image(patches_list, cord_list, max_row, max_col):
    max_row, max_col =  cord_list[-1]
    fig, ax = plt.subplots(max_row+1, max_col+1, figsize=(25, 25))
    for idx, image_patch in enumerate(cord_list):
        image_patch = patches_list[idx]
        # print(idx)
        x, y = cord_list[idx]
        ax[x][y].imshow(image_patch)

# %%
show_patch_image(patches_list, cord_list, max_row, max_col)

# %%
fig, ax = plt.subplots(max_row+1, max_col+1, figsize=(25, 25))
for idx, image_patch in enumerate(cord_list):
    image_patch = patches_list[idx]
    # print(idx)
    x, y = cord_list[idx]
    ax[x][y].imshow(image_patch)

# %%
a = np.array(cord_list)
print(a[-1])

# %%
input = Input(shape=(INPUT_SHAPE))


