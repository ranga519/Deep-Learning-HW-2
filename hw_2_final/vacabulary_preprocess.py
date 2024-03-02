import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch import nn
from torchvision import models
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
import os
import cv2
import sys
import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

tl=time.localtime()
format_time = time.strftime("%Y%m%d_%H%M%S", tl) 

def build_vocab(trainset,testset,word_count_threshold, unk_requried=False):
    dataset_all = trainset+testset
    sentance_all = []
    for i in range(len(dataset_all)):
        
        sub_sentance = dataset_all[i]['caption']
        for j in sub_sentance:
            sentance_all.append(j)
    all_captions = []
    word_counts = {}
    for cap in sentance_all:
        caption = cap.split('\t')[-1] 
        caption = '<BOS> ' + caption + ' <EOS>'
        all_captions.append(caption)
        for word in caption.split(' '):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        for word in word_counts:
            if word_counts[word] < word_count_threshold:
                word_counts.pop(word)
                unk_requried = True
    return word_counts, unk_requried


def word_to_ids(word_counts, unk_requried):
    word_to_id = {}
    id_to_word = {}
    count = 0
    if unk_requried:
        word_to_id['<UNK>'] = count
        id_to_word[count] = '<UNK>'
        count += 1
        print("<UNK> True")
    for word in word_counts:
        word_to_id[word] = count
        id_to_word[count] = word
        count += 1
    return word_to_id, id_to_word



# convert each word of captions to the index
def convert_caption(captions, word_to_id, max_length):
    if type(captions) == 'str':
        captions = [captions]
    caps, cap_mask = [], []
    for cap in captions:
        nWord = len(cap.split(' '))
        cap = cap + ' <EOS>'*(max_length-nWord)
        cap_mask.append([1.0]*nWord + [0.0]*(max_length-nWord))
        cap_ids = []
        for word in cap.split(' '):
            if word in word_to_id:
                cap_ids.append(word_to_id[word])
            else:
                cap_ids.append(word_to_id['<UNK>'])
        caps.append(cap_ids)
    return np.array(caps), np.array(cap_mask)



# fetch features of train videos
def get_train_data(batch_size):
    vid = np.random.choice(video_train, batch_size)
    # url = [Vid2Url[video] for video in vid]
    cur_vid = np.array([np.load(VIDEO_DIR_TRAIN+video+'.npy') for video in vid])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_train[video], 1)[0] for video in vid]
    captions_list = captions
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask, captions_list, vid


# fetch features of test videos
def get_test_data(batch_size):
    vid = np.random.choice(video_test, batch_size)
    # url = [Vid2Url[video] for video in vid]
    cur_vid = np.array([np.load(VIDEO_DIR_TEST+video+'.npy') for video in vid])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_test[video], 1)[0] for video in vid]
    captions_list = captions
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask, captions_list, vid



# print captions
def print_in_english(captions):
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    
    cap_list = []
    for cap in captions_english:
        if '<EOS>' in cap:
            cap = cap[0:cap.index('<EOS>')]
        # print(' ' + ' '.join(cap))
        cap_list.append((' ' + ' '.join(cap)))
    return cap_list



def get_frame(video_name):
    testvideo = cv2.VideoCapture(video_name)
    if testvideo.isOpened():
        frames = int(testvideo.get(cv2.CAP_PROP_FRAME_COUNT))
        width = testvideo.get(cv2.CAP_PROP_FRAME_WIDTH) 
        height = testvideo.get(cv2.CAP_PROP_FRAME_HEIGHT) 
        frame = 0
        while 1:
            if frame == 15:
                ret, frame_img = testvideo.read()
                # print('frame captured')
                break
            else:
                frame = frame+1
    return frame_img



# only works when batch_size = 10
def batch_train_vis(vid):
    frames_1 = np.array([])
    for i in range(0,5):
        frame_i = get_frame(VIDEO_TRAIN+vid[i])
        frame_i = cv2.resize(frame_i, (128,128))
        if i == 0:
            frames_1 = frame_i
        else:
            frames_1 = np.vstack((frames_1,frame_i))
            
    frames_2 = np.array([])            
    for i in range(5,10):
        frame_i = get_frame(VIDEO_TRAIN+vid[i])
        frame_i = cv2.resize(frame_i, (128,128))
        if i == 5:
            frames_2 = frame_i
        else:
            frames_2 = np.vstack((frames_2,frame_i))
        
    frames = np.hstack((frames_1,frames_2))
    cell = Image.fromarray(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
    return frames, cell

#####################################################################################

def batch_test_vis(vid):
    frames_1 = np.array([])
    for i in range(0,5):
        frame_i = get_frame(VIDEO_TEST+vid[i])
        frame_i = cv2.resize(frame_i, (128,128))
        if i == 0:
            frames_1 = frame_i
        else:
            frames_1 = np.vstack((frames_1,frame_i))
            
    frames_2 = np.array([])            
    for i in range(5,10):
        frame_i = get_frame(VIDEO_TEST+vid[i])
        frame_i = cv2.resize(frame_i, (128,128))
        if i == 5:
            frames_2 = frame_i
        else:
            frames_2 = np.vstack((frames_2,frame_i))
        
    frames = np.hstack((frames_1,frames_2))
    cell = Image.fromarray(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
    return frames, cell

#####################################################################################

# save training log
def write_txt(epoch, iteration, loss, time_cost):
    with open("./save_text/train_log_%s.txt"%(format_time), 'a') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\t time_for_completion:[ %f ]\n" % (epoch, iteration, loss, time_cost))
        
#####################################################################################

def output_txt(filename, video_name, captions):
    with open("./%s"%(filename), 'a') as f:
        f.write("%s,%s\n" % (video_name, captions[0]))

#####################################################################################
