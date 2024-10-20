# %%
import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate , LeakyReLU, BatchNormalization , ReLU, Embedding
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
def show_img(img_np):
    plt.imshow(img_np)
    plt.show()
def show_spaning_cell(img_np, table_spaning_cell):
    for box in table_spaning_cell:
        x1 , y1 , x2 , y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_np
def show_img_box_raw(img_np, box_list_row, box_list_col,table_spaning_cell = []):
    img_raw = img_np.copy()
    img_span_cell = show_spaning_cell(img_raw, table_spaning_cell)
    for box in box_list_row:
        x1 , y1 , x2 , y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for box in box_list_col:
        x1 , y1 , x2 , y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    fig, ax = plt.subplots(1, 2, figsize=(25, 25))
    ax[0].imshow(img_raw)
    ax[1].imshow(img_np)

# %%
img_path  = data_path + file_name + '.jpg'
image = Image.open(img_path)
image_np = np.array(image)
image_np = cv2.resize(image_np, (256,256))
image_np = np.array(image_np)

cell_list, cord_list = get_cell_list(row, col)
img_np = show_spaning_cell(image_np, cell_list)
plt.imshow(img_np)
plt.show()

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
max_row, max_col =  cord_list[-1]

# %%
show_patch_image(patches_list, cord_list, max_row, max_col)

# %%
INPUT_SHAPE

# %%
class ROIPooling(tf.keras.layers.Layer):   # v2
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(ROIPooling, self).__init__(**kwargs)

    def call(self, feature_map, rois):
        """
        Args:
            feature_map: Tensor of shape (batch_size, height, width, channels)
            rois: Tensor of shape (1, num_rois, 4) with each ROI represented as
                  [x1, y1, x2, y2]

        Returns:
            Pooled features of shape (num_rois, pool_size, pool_size, channels)
        """
        # print(rois.shape)
        num_rois = 10000
        # print(f'num_rois {num_rois}')
        b,h,w,c = feature_map.shape
        outputs = []
        for roi_idx in range(num_rois):
            x1 = rois[0, roi_idx, 0]
            y1 = rois[0, roi_idx, 1]
            x2 = rois[0, roi_idx, 2]
            y2 = rois[0, roi_idx, 3]

            x1 = K.cast(x1, 'int32')
            y1 = K.cast(y1, 'int32')
            x2 = K.cast(x2, 'int32')
            y2 = K.cast(y2, 'int32')
            if (x1 ==0) and (y1 ==0 ) and (x2 ==0) and (y2 ==0):
                rs = tf.zeros(shape= (1, self.pool_size[0], self.pool_size[1], c))
            else :
                rs = tf.image.resize(feature_map[:, y1:y2, x1:x2, :], self.pool_size)
            outputs.append(rs)
        final_output = K.concatenate(outputs, axis=0)
        print(final_output.shape)
        pooled_features = K.reshape(final_output, (1, num_rois, self.pool_size[0], self.pool_size[1], -1))
        return pooled_features
    def get_config(self):
        config = super().get_config()
        config.update({'pool_size': self.pool_size})
        return config

# %%
x_position_embedding = tf.range(start=0, limit=100, delta=1)
y_position_embedding = tf.range(start=0, limit=100, delta=1)

# %%
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.num_dims  = projection_dims

        self.dense = tf.keras.layers.Dense(units = projection_dims)
        self.x_positional_embeddings = Embedding(input_dim = num_patches, output_dim = projection_dims)
        self.x_positional_embeddings = Embedding(input_dim = num_patches, output_dim = projection_dims)

    def call(self, x):
        x_position = tf.range(0,limit=self.num_patches, delta=1)
        y_position = tf.range(0,limit=self.num_patches, delta=1)

        encoded = self.dense(x) + self.x_positional_embeddings(x_position) + self.y_positional_embeddings(y_position)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_paches": self.num_patches,
            "d": self.num_dims,
        })
        return config

# %%
INPUT_SHAPE =(1024,1024,3)
input_tensor = Input(shape=(INPUT_SHAPE))
input_cell_box = Input(shape=(100*100,4)) # add 0,0,0,0 cho du 100*100 ( handle during loading data)
input_cord = Input(shape=(100*100,2)) # add 0,0 cho du 100*100 ( handle during loading data) # using for positon embedding 
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

p2_2 = Conv2D(128, kernel_size = 3, strides = 1, padding ='same')(p2)
print(p2_2.shape)
# input_cell_box = tf.squeeze(input_cell_box, axis=0)
crops = ROIPooling(pool_size=(7, 7))(p2_2, input_cell_box)

#position embedded ing 
embedded_patches= tf.reshape(crops, (10000, -1))
# embedded_patches = tf.keras.layers.Dense(512)(crops)

out = p2_2

model = Model(inputs = input_tensor, outputs = out)

# %%
crops.shape

# %%
embedded_patches.shape

# %%
crops.shape

# %%
model.output_shape

# %%
model.summary()


