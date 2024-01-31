# Frmae_loader is for extraction video frames from csv file which made by TTset_parser or filtered by Data_filter #
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import cv2 as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class frame_loader (Dataset): # Format of Pytorch #
    def __init__(self, all_data, action_filter, transform = None, target_transform = None):
        self.all_data = all_data
        self.transform = transform
        self.target_transform = target_transform
        self.action_filter = action_filter

    def __len__(self): 
        return len(self.all_data) # Length of data

    def __getitem__(self, idx):
        max_frame = 60 # Extract 60 frames per video
        data = get_clips(self.all_data[idx][0],  self.all_data[idx][2], self.all_data[idx][3], self.action_filter, max_frame, CC=True, opt_flow=False) 
        label = self.all_data[idx][1]  # Get label
        label = int(label)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

def get_clips(clip_path, frames_idx, view_type, action_filter_require, max_frame, CC, opt_flow) -> Tuple[Tensor, int]:
    f_idx = False
    data = []
    tem = []
    zero_image = []
    clip = []
    count_constrain = 0
    if action_filter_require == True:
        frames_idx_counter = 0
        frames_idx = int(frames_idx)
    view_type = int(view_type)
    data = torch.FloatTensor(data) 
    cap = cv.VideoCapture(clip_path) # Capture video
    frame_number = 0
    while(1):
        ret, frame = cap.read() #讀取影片時會回傳兩個參數，ret = return，代表有無成功讀取，frame就是當下的偵，每read一次讀一偵，要讀影片的話就需要使用while迴圈
                                # ret = return, when return = 0, means video is over
                                # frame = precent frame of video, use fuction while(1) to keep capturing video         
        if not ret:
            break
        if f_idx == True: # If user set cut video is True
            frames_idx_counter = frames_idx_counter + 1
            if frames_idx_counter > frames_idx:
                frames_idx_counter = 0
                break
        if CC == True: # CC = module of image crop, there has verison 1 and version 2
            frame = image_crop(frame, view_type)
        frame = cv.resize(frame, (230, 230)) # Resize frame resolution into 230 x 230
        clip.append(frame)
    cap.release()

# If user want to use module of fuzzy rules, then need to activate Optical-flow
# --------------------------- Optical-flow -------------------------------
    # if opt_flow == True:
    #     cut_point = Dense_optical_flow(clip)
    #     try:
    #         for i in range (cut_point):
    #             clip.pop(0)
    #         # print(len(clip))
    #     except:
    #         print("Pop error!!!")
    #         print(len(clip), end="")
    #         print(", ", end="")
    #         print(cut_point)
    #         print(i)
    #         print("----------------")
# --------------------------- Optical-flow -------------------------------

# Video Length Adaptive Module (VLAM), to average capture video frames
# ---------------------------VLAM-------------------------------
    frame_number = (int(len(clip))/max_frame)
    frame_number = int(frame_number)
    if frame_number == 0:
        for i in range (len(clip)):
            j = clip[i]
            j = torch.FloatTensor(j)
            tem.append(torch.unsqueeze(j, 0))
        while(1):
            if len(tem) == max_frame:
                break
            zero_image = torch.zeros((230, 230, 3), dtype=torch.float)
            zero_image = torch.FloatTensor(zero_image)
            tem.append(torch.unsqueeze(zero_image, 0))
            zero_image = []
    if frame_number > 0:
        for i in range (len(clip)):
            if count_constrain % frame_number == 0: #每frame_number偵取一偵，取決於影片長度
                j = clip[i] #將圖片都固定解析度
                j = torch.FloatTensor(j)
                tem.append(torch.unsqueeze(j, 0))
                if len(tem) == max_frame: #如果圖像超過儲存上限，結束迴圈
                    break
            count_constrain = count_constrain + 1 #計時器加一
# ---------------------------VLAM-------------------------------
    clip = []
    data = torch.cat(tem)
    data = data.permute(3, 0, 1, 2)
    # print("Data shape: {}".format(data.shape))
    return data

def image_crop(frame, view_type):
    ## Image crop v1 ##
    w = frame.shape[1]
    h = frame.shape[0]
    TTset_w = int(w/5) 
    TTset_h = int(h/8) 
    crop_img_quarter = frame[TTset_h:(h-TTset_h), TTset_w:(w-TTset_w)]  # Cut 1/5 of left side and 1/5 of right side
                                                                        # Cut 1/8 of top and 1/8 of button

    ## Image crop v2 ##
    # if int(view_type) == 1: 
    #     w = frame.shape[1]
    #     h = frame.shape[0]
    #     TTset_w = int(w/4) + 100
    #     TTset_h = int(h/6)
    #     crop_img_quarter = frame[TTset_h:(h-TTset_h), TTset_w:(w-TTset_w)]
    # if int(view_type) == 2: 
    #     w = frame.shape[1]
    #     h = frame.shape[0]
    #     TTset_w = int(w/6)
    #     TTset_h = int(h/6)
    #     crop_img_quarter = frame[TTset_h:h-TTset_h, TTset_w:w-TTset_w]
    # if int(view_type) == 3: 
    #     w = frame.shape[1]
    #     h = frame.shape[0]
    #     TTset_w = int(w/5)
    #     TTset_h = int(h/5)
    #     crop_img_quarter = frame[TTset_h + 50:((h+50)-TTset_h), TTset_w:(w-TTset_w)]
    # if int(view_type) == 4: 
    #     w = frame.shape[1]
    #     h = frame.shape[0]
    #     TTset_w = int(w/5) + 30
    #     TTset_h = int(h/6)
    #     crop_img_quarter = frame[TTset_h:(h-TTset_h), TTset_w:(w-TTset_w)]
    return crop_img_quarter

def Dense_optical_flow(clip):
    cou = 0
    frame1 = clip[0]
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY) # Transfer RGB image to gray image
    hsv = np.zeros_like(frame1) # Creat a zero matrix, which has same resolution with video frames 
    hsv[..., 1] = 255 # use 0 ~ 255 to present brightness of pixel
    for i in range (len(clip)):
        frame2 = clip[i]
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        s = sum(hsv)
        s = sum(s)
        s = sum(s)
        if s > 350: 
            break
        cou = cou + 1
    return cou
