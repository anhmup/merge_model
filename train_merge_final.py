# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %%
H_SHAPE = 768
W_SHAPE = 512

# %%
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
import cv2
import ast
from tqdm import  tqdm

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %%

import glob
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# %% [markdown]
# # model

# %%
class Resnet_FPN(nn.Module):
    def __init__(self,out_channels = 256):
        super(Resnet_FPN, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.to(device)
        self.layer_1 = nn.Sequential(*list(self.resnet.children())[:4])
        self.layer_2 = self.resnet.layer2 
        self.layer_3 = self.resnet.layer3
        self.layer_4 = self.resnet.layer4

        self.lateral4 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if (torch.isnan(x).any()):
            print("error at Resnet_FPN")
        c1 = self.layer_1(x)
        if (torch.any(torch.isnan(c1))):
            print('error resnet ')
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')

        p4 = self.output4(p4)
        p3 = self.output3(p3)
        p2 = self.output2(p2)
        p1 = self.output1(p1)

        return p1, p2, p3, p4

# %%
def convert_rois_to_boxes(rois):
    MxN = (torch.max(rois[:,0])+1)*(torch.max(rois[:,1])+1).item()
    MxN = int(MxN)
    batch_index = torch.zeros(MxN)
    x1 = rois[:,2]
    y1 = rois[:,3]
    x2 = rois[:,4]
    y2 = rois[:,5]

    boxes = torch.stack([batch_index,x1,y1,x2,y2],dim=1)
    return boxes

# %%
class convert_rois_to_box(nn.Module):
    def __init__(self):
        super(convert_rois_to_box, self).__init__()
        # pass
    def forward(self, rois):
        MxN = (torch.max(rois[:,0])+1)*(torch.max(rois[:,1])+1).item()
        MxN = int(MxN)
        batch_index = torch.zeros(MxN)

        batch_index = batch_index.to(device)
        x1 = rois[:,2]
        y1 = rois[:,3]
        x2 = rois[:,4]
        y2 = rois[:,5]

        boxes = torch.stack([batch_index,x1,y1,x2,y2],dim=1)
        return boxes

# %%
class position_embedding(nn.Module):
    def __init__(self, num_paches = 100, projection_dims = 256):
        super(position_embedding, self).__init__()
        self.x_position_embeddings = nn.Embedding(num_paches, projection_dims)
        nn.init.constant_(self.x_position_embeddings.weight, 0.)
        self.y_position_embeddings = nn.Embedding(num_paches, projection_dims)
        nn.init.constant_(self.y_position_embeddings.weight, 0.)
    def forward(self, rois):
        x = torch.max(rois[:,1])
        y = torch.max(rois[:,0])

        x = x.to(torch.int32)
        y = y.to(torch.int32)
        x_1 = x+1
        y_1 = y+1
        
        col = torch.arange(x+1)
        col = col.to(device)
        row = torch.arange(y+1)
        row = row.to(device)

        x_pos = col.repeat(y_1)
        y_pos = row.repeat_interleave(x_1)

        x_embed = self.x_position_embeddings(x_pos)
        y_embed = self.y_position_embeddings(y_pos)
        
        return x_embed, y_embed

# %%
class predict_head_row(nn.Module):
    def __init__(self,hidden_dim = 512):
        super(predict_head_row, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff1 = nn.Linear(in_features=512, out_features=512)
        self.ff2 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.ff1(x)
        x = self.relu(x)
        x= self.ff2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x
class predict_head_col(nn.Module):
    def __init__(self,hidden_dim = 512):
        super(predict_head_col, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff1 = nn.Linear(in_features=512, out_features=512)
        self.ff2 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.ff1(x)
        x = self.relu(x)
        x= self.ff2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x

# %%
def roi_align(feature_map, rois, output_size = (7,7)):
    return ops.roi_align(feature_map, rois, output_size , spatial_scale=1/4)

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim = 512, num_layers = 3, num_heads = 8, ff_dim = 512):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_dim = input_dim

    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        output = self.transformer_encoder(src)
        return output

# %%
class Model_final(nn.Module):
    def __init__(self, num_layers = 3 ):
        super(Model_final, self).__init__()
        self.pos_embedding = position_embedding()
        self.convert_rois_to_box = convert_rois_to_box()
        self.encoder = TransformerEncoder()
        self.backbone = Resnet_FPN()

        # self.merge_head = merge_head()
        self.row_head = predict_head_row()
        self.col_head = predict_head_col()
        self.middle = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.flatten_fm = nn.Linear(in_features= 6272, out_features= 512)
    def forward(self, x , rois):
        if (torch.isnan(x).any()):
            print("error fw model ")
        rois_1 = torch.tensor(rois[0])
        rois_new = self.convert_rois_to_box(rois_1)
        x_embed, y_embed = self.pos_embedding(rois[0])
        if (torch.any(torch.isnan(x_embed))):
            print('pos error')
        elif  (torch.any(torch.isnan(y_embed))):
            print('pos error')
        pos_  = torch.cat((x_embed, y_embed), dim=-1) # (1, MxN, 512)
        p1, p2, p3, p4 = self.backbone(x) # p1: (1, 256, 128, 128)
        if  (torch.any(torch.isnan(p1))):
            print('feature error')
        feature_map = self.middle(p1) #  (1, 256, 128, 128)
        # print(f'feature_map :{feature_map.shape}')
        crops = roi_align(feature_map, rois_new) # (MxN, 128, 7, 7)  
        if  (torch.any(torch.isnan(crops))):
            print(' roi align error ')
        embedded_patches = crops.reshape(-1, 128*7*7) # (MxN, 6272)
        encoded_patches = self.flatten_fm(embedded_patches)# (MxN, 512)
        encoded_patches = torch.unsqueeze(encoded_patches, dim=0) # (1, MxN, 512)
        encode = pos_ + encoded_patches # (1, MxN , 512)
        # encode = encoded_patches

        x = self.encoder(encode) # (1, MxN, 512)
        if  (torch.any(torch.isnan(x))):
            print(' encoder error ')        
        row_logits = self.row_head(x)
        col_logits = self.col_head(x)
        col_logits = col_logits.view(-1)
        row_logits = row_logits.view(-1)
        return row_logits , col_logits

# %% [markdown]
# # data loader

# %% [markdown]
# # data loader

# %%
matrix_col_path  = 'C:/Users/SEHC/Desktop/qa/LV/Merge/data_train_full/data/col/'
matrix_row_path  = 'C:/Users/SEHC/Desktop/qa/LV/Merge/data_train_full/data/row/'
image_path = 'C:/Users/SEHC/Desktop/qa/LV/Merge/data_train_full/data/image/'

# %%
def convert_String_to_array(string_array):
    nested_list = ast.literal_eval(string_array)

    return nested_list

def normalize_img(img):
    #norm_img = (img - img.min()) / (img.max() - img.min())
    norm_img = np.array(img,dtype= np.float32)/255.0
    return norm_img

# %%
# class CustomDataset(Dataset):
#     def __init__(self, data_frame):
#         self.data = data_frame
#     def __len__(self):
#         return len(self.data)
#     def load_gt(self, image_name):
#         row_gt = np.load(matrix_row_path + image_name + '.npy')
#         row_gt_1 = np.reshape(row_gt, -1)
#         col_gt = np.load(matrix_col_path + image_name + '.npy')
#         col_gt_1 = np.reshape(col_gt, -1)
#         return row_gt_1, col_gt_1
#     def load_image(self, image_name):
#         img_path  = image_path + image_name + '.jpg'
#         image = Image.open(img_path)
#         image_np = np.array(image)
#         image_input = cv2.resize(image_np, (512,512))
#         image_input = image_input.astype('float')
#         image_input = normalize_img(image_input)
#         return image_input
#     def __getitem__(self, idx):
#         image_name = self.data.iloc[idx][0]
#         # print(image_name)
#         bbox = self.data.iloc[idx][1]
#         image = self.load_image(image_name)

#         image_tensor = torch.from_numpy(image)
#         image_tensor = image_tensor.permute(2,0,1) # C , H , W
#         cell_bbox = np.array(convert_String_to_array(bbox))
#         # cell_bbox = np.squeeze(cell_bbox, axis= 0)
#         row_gt , col_gt = self.load_gt(image_name)
#         return {'image': image_tensor, 'bbox' : cell_bbox}, {'row' : row_gt,'col' : col_gt}

# %%
class CustomDataset(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
    def __len__(self):
        return len(self.data)
    def load_gt(self, image_name):
        row_gt = np.load(matrix_row_path + image_name + '.npy')
        row_gt_1 = np.reshape(row_gt, -1)
        col_gt = np.load(matrix_col_path + image_name + '.npy')
        col_gt_1 = np.reshape(col_gt, -1)
        return row_gt_1, col_gt_1
    def load_image(self, image_name):
        img_path  = image_path + image_name + '.jpg'
        image = Image.open(img_path)
        image_np = np.array(image)
        image_input = cv2.resize(image_np, (W_SHAPE,H_SHAPE))
        image_input = image_input.astype('float')
        image_input = normalize_img(image_input)
        return image_input
    def convert_to_target_shape(self, original_image, cell_bbox_list, shape = (H_SHAPE,W_SHAPE)):
        h,w,c = np.array(original_image).shape
        cell_bbox_target_list = []
        for bbox_cell in cell_bbox_list:
            idx_row, idx_col, x1, y1 , x2, y2 = bbox_cell
            x1_target = int(x1/512*W_SHAPE)
            y1_target = int(y1/512*H_SHAPE)
            x2_target = int(x2/512*W_SHAPE)
            y2_target = int(y2/512*H_SHAPE)
            bbox_cell_target  = [idx_row, idx_col, x1_target, y1_target, x2_target, y2_target]
            cell_bbox_target_list.append(bbox_cell_target)
        return cell_bbox_target_list
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx][0]
        # print(image_name)
        bbox = self.data.iloc[idx][1]
        image = self.load_image(image_name)

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1) # C , H , W
        cell_bbox_list = convert_String_to_array(bbox)
        bbox_target_list = self.convert_to_target_shape(image, cell_bbox_list, shape = (H_SHAPE,W_SHAPE))
        cell_bbox = np.array(bbox_target_list)
        # cell_bbox = np.squeeze(cell_bbox, axis= 0)
        row_gt , col_gt = self.load_gt(image_name)
        return {'image': image_tensor, 'bbox' : cell_bbox}, {'row' : row_gt,'col' : col_gt}

# %%
df_test = pd.read_csv('fintab_merge_data_train_full.csv')
data_test = CustomDataset(df_test[:1])
data_loader = DataLoader(data_test, batch_size= 1, shuffle=False )

# %%
model_test = Model_final()
model_test.to(device)
input_ , output_ = next(iter(data_loader))

# %%
test_image = input_['image'][0]
test_bbox = input_['bbox'][0]
test_image =   test_image.permute(1,2,0)
test_image_np = np.array(test_image)

img_test_draw = test_image_np.copy()
bbox = test_bbox.numpy()

for bbox_cell in bbox:
    idx_row, idx_col, x1, y1 , x2, y2 = bbox_cell
    cv2.rectangle(img_test_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(img_test_draw)
plt.show()

# %%
img_in_test = input_['image']
bbox_in_test = input_['bbox']
img_in_test = img_in_test.to(device)
bbox_in_test = bbox_in_test.to(device)
row_out_test, col_out_test = model_test(img_in_test, bbox_in_test)

# %%
output_

# %%
output_['col']

# %% [markdown]
# # train

# %%
data_frame = pd.read_csv('fintab_merge_data_train_full.csv')
print(len(data_frame))
data_frame = data_frame[:80000]
data = CustomDataset(data_frame)
data_loader = DataLoader(data, batch_size=1, shuffle=True)

# %%
device

# %%
model_1 = Model_final()
# model_1 = nn.DataParallel(model_1)
model_1.to(device)
# weights_merge = torch.load("merge_model_2.pth", map_location= device)
# model_1.load_state_dict(weights_merge)
# convert_rois_to_boxes = convert_rois_to_boxes.to(device)
optimizer = optim.AdamW(model_1.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

# %%
def check_for_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN values found in parameter: {name}")
            return True
    return False

# %%
def save_check_point(epoch, model_1 , optimizer):
    folder_checkpoint = 'C:/Users/SEHC/Desktop/qa/LV/Merge/check_point/'
    checkpoint = {
            'epoch': epoch ,
            'state_dict': model_1.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    torch.save(checkpoint, folder_checkpoint + 'check_point_merge_model_fintab_80k' + str(epoch) +'.pth')

# %%
num_epochs = 25
loss_list  =[]
loss_row_list = []
loss_col_list = []
for epoch in range(num_epochs):
    for i, (inputs, gt) in tqdm(enumerate(data_loader)):
        # print(f'epoch :{epoch}, index {i}')
        # model_1.train(True)
        img_in = inputs['image']
        bbox_in = inputs['bbox']

        row_gt = gt['row']
        row_gt = row_gt.float()
        row_gt = row_gt.view(-1)
        col_gt = gt['col']
        col_gt = col_gt.float()
        col_gt = col_gt.view(-1)
        img_in = img_in.to(device)
        if (torch.isnan(img_in).any()):
            print("error")
            continue
        bbox_in = bbox_in.to(device)
        row_gt = row_gt.to(device)
        col_gt = col_gt.to(device)
        optimizer.zero_grad()
        row_out, col_out = model_1(img_in, bbox_in)
        # if not ((row_out < 1).all()):
        #     print(row_out)
        #     break
        # if not ((col_out < 1).all()):
        #     print(col_out)
        #     # break
        row_loss = nn.BCELoss()(row_out, row_gt)
        col_loss = nn.BCELoss()(col_out, col_gt)
        total_loss = row_loss+ col_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), 0.001)
        optimizer.step()
        if check_for_nan(model_1):
            print("NaN values detected in model parameters. Stopping training.")
            # break
        
    #     loss_list.append(total_loss.item())
    #     loss_col_list.append(col_loss.item())
    #     loss_row_list.append(row_loss.item())
        del row_loss
        del col_loss
        del  total_loss
        del row_out
        del col_out
    # loss_epoch = sum(loss_list)/len(loss_list)
    # loss_col_epoch = sum(loss_col_list)/len(loss_col_list)
    # loss_row_epoch = sum(loss_row_list)/len(loss_row_list)
    # print(f'epoch:{epoch}============ loss: {loss_epoch}  col :{loss_col_epoch}   row:{loss_row_epoch}')  
    # loss_list.clear()   
    # loss_col_list.clear()
    # loss_row_list.clear()
    # save check point 
    torch.save(model_1.state_dict(), "merge_model_fintab_80_768_512_epoch_" + str(epoch) + ".pth")


# %%
num_epochs = 1
loss_list  =[]
loss_row_list = []
loss_col_list = []
for epoch in range(25,26):
    for i, (inputs, gt) in tqdm(enumerate(data_loader)):
        # print(f'epoch :{epoch}, index {i}')
        # model_1.train(True)
        img_in = inputs['image']
        bbox_in = inputs['bbox']

        row_gt = gt['row']
        row_gt = row_gt.float()
        row_gt = row_gt.view(-1)
        col_gt = gt['col']
        col_gt = col_gt.float()
        col_gt = col_gt.view(-1)
        img_in = img_in.to(device)
        if (torch.isnan(img_in).any()):
            print("error")
            continue
        bbox_in = bbox_in.to(device)
        row_gt = row_gt.to(device)
        col_gt = col_gt.to(device)
        optimizer.zero_grad()
        row_out, col_out = model_1(img_in, bbox_in)
        # if not ((row_out < 1).all()):
        #     print(row_out)
        #     break
        # if not ((col_out < 1).all()):
        #     print(col_out)
        #     # break
        row_loss = nn.BCELoss()(row_out, row_gt)
        col_loss = nn.BCELoss()(col_out, col_gt)
        total_loss = row_loss+ col_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_1.parameters(), 0.001)
        optimizer.step()
        if check_for_nan(model_1):
            print("NaN values detected in model parameters. Stopping training.")
            # break
        
    #     loss_list.append(total_loss.item())
    #     loss_col_list.append(col_loss.item())
    #     loss_row_list.append(row_loss.item())
        del row_loss
        del col_loss
        del  total_loss
        del row_out
        del col_out
    # loss_epoch = sum(loss_list)/len(loss_list)
    # loss_col_epoch = sum(loss_col_list)/len(loss_col_list)
    # loss_row_epoch = sum(loss_row_list)/len(loss_row_list)
    # print(f'epoch:{epoch}============ loss: {loss_epoch}  col :{loss_col_epoch}   row:{loss_row_epoch}')  
    # loss_list.clear()   
    # loss_col_list.clear()
    # loss_row_list.clear()
    # save check point 
    torch.save(model_1.state_dict(), "merge_model_fintab_80_768_512_epoch_" + str(epoch) + ".pth")
    # if epoch == 1 :
    #     torch.save(model_1.state_dict(), "merge_model_fintab_80_1.pth")
    #     save_check_point(epoch, model_1, optimizer)
    # if epoch == 5:
    #     torch.save(model_1.state_dict(), "merge_model_fintab_80_5.pth")
    #     save_check_point(epoch, model_1, optimizer)
    # if epoch == 10:
    #     torch.save(model_1.state_dict(), "merge_model_fintab_80_10.pth")
    #     save_check_point(epoch, model_1, optimizer)    
    # if epoch == 15:
    #     torch.save(model_1.state_dict(), "merge_model_fintab_80_15.pth")
    #     save_check_point(epoch, model_1, optimizer)
    # if epoch == 20:
    #     torch.save(model_1.state_dict(), "merge_model_fintab_80_20.pth")
    #     save_check_point(epoch, model_1, optimizer)

# %%
row_out.shape

# %%
row_gt.shape

# %%
bbox_in.shape

# %%
gt['row'].shape

# %%
gt['col'].shape

# %% [markdown]
# # test 

# %%
torch.save(model_1.state_dict(), "merge_model_2.pth")

# %%
model_after = Model_final()
model_after.to(device)
weights = torch.load('merge_model_2.pth', map_location = device)
model_after.load_state_dict(weights)

# %%
i = 31008

# %%
df_test = pd.read_csv('test_4.csv')
i= i +1
data_test = CustomDataset(df_test[i:i+1])
data_loader = DataLoader(data_test, batch_size= 1, shuffle=False )
input_ , output_ = next(iter(data_loader))
input_image = input_['image'].to(device)
input_bbox =  input_['bbox'].to(device)
pred_row, pred_col = model_after(input_image , input_bbox)

# %%
output_['col']

# %%
pred_col

# %%
last_cell =  input_['bbox'].numpy()[-1]
num_row, num_col =  last_cell[-1][0] , last_cell[-1][1] 
print(num_row , num_col)

# %%
pred_col = pred_col.cpu()
mask_1 = pred_col > 0.9
mask_0 = pred_col <=0.9
final_pred_col = np.zeros(shape = np.shape(pred_col) , dtype= int)
final_pred_col[np.where(mask_1 == True)] = 1 
final_pred_col = np.reshape(final_pred_col, (num_row + 1, num_col + 1))
print(final_pred_col)

# %%
print(np.reshape(output_['col'].numpy(), (num_row +1, num_col +1 )))

# %%
test_image = input_['image'][0]
test_bbox = input_['bbox'][0]
test_image =   test_image.permute(1,2,0)
test_image_np = np.array(test_image)

img_test_draw = test_image_np.copy()
bbox = test_bbox.numpy()

for bbox_cell in bbox:
    idx_row, idx_col, x1, y1 , x2, y2 = bbox_cell
    cv2.rectangle(img_test_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(img_test_draw)
plt.show()


