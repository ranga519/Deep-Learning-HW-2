
import os
import cv2
import sys
import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models



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
    cur_vid = np.array([np.load(VIDEO_DIR_TEST+video+'.npy') for video in vid])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(Vid2Cap_test[video], 1)[0] for video in vid]
    captions_list = captions
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask, captions_list, vid



# printing captions
def print_in_english(captions):
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    
    cap_list = []
    for cap in captions_english:
        if '<EOS>' in cap:
            cap = cap[0:cap.index('<EOS>')]
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



# save training log
def write_txt(epoch, iteration, loss, time_cost):
    with open("./save_text/train_log_%s.txt"%(format_time), 'a') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\t time_for_completion:[ %f ]\n" % (epoch, iteration, loss, time_cost))
        

def output_txt(filename, video_name, captions):
    with open("./%s"%(filename), 'a') as f:
        f.write("%s,%s\n" % (video_name, captions[0]))



class S2VT(nn.Module):
    def __init__(self, vocab_size, batch_size=15, frame_dim=4096, hidden=600, dropout=0.6, n_step=80):
        super(S2VT, self).__init__()
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden = hidden
        self.n_step = n_step

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden)
        self.linear2 = nn.Linear(hidden, vocab_size)

        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(2*hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, video, caption=None):
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video)                   
        video = video.view(-1, self.n_step, self.hidden)
        padding = torch.zeros([self.batch_size, self.n_step-1, self.hidden]).cuda()
        video = torch.cat((video, padding), 1)        
        vid_out, state_vid = self.lstm1(video)

        if self.training:
            caption = self.embedding(caption[:, 0:self.n_step-1])
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            caption = torch.cat((padding, caption), 1)        
            caption = torch.cat((caption, vid_out), 2)       

            cap_out, state_cap = self.lstm2(caption)
            cap_out = cap_out[:, self.n_step:, :]
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            return cap_out
        else:
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            cap_input = torch.cat((padding, vid_out[:, 0:self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input)
            
            bos_id = word2id['<BOS>']*torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)

            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)

            caption = []
            caption.append(cap_out)
            for i in range(self.n_step-2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                caption.append(cap_out)
            return caption


if __name__ == "__main__":
    time_start_all = time.time()
    time_start = time.time()

    if len(sys.argv)>=2:
        data_path = './'+str(sys.argv[1])+'/'
    else:
        data_path = './MLDS_hw2_1_data/'

    if len(sys.argv)>=3:
        output_filename = str(sys.argv[2])
    else:
        output_filename = 'test_output.txt'
    print('load successful...')
    print('data_path = ', data_path)
    print('output_filename = ', output_filename)
        
    n_lstm_steps = 80
    model_path = './'
    model_file = 's2vt_best.pth'
    VIDEO_TRAIN = data_path+'training_data/video/'
    VIDEO_TEST  = data_path+'testing_data/video/'

    VIDEO_DIR_TRAIN = data_path+'training_data/feat/'
    VIDEO_DIR_TEST  = data_path+'testing_data/feat/'
    
    trainset = json.load(open(data_path+'training_label.json','r'))
    testset  = json.load(open(data_path+'testing_label.json','r'))
    
    video_train = list(open(data_path+'training_data/id.txt','r').read().splitlines())
    video_test  = list(open(data_path+'testing_data/id.txt','r').read().splitlines())
    
        
    # build vocab
    Vid2Cap_train = {}
    for i in range(len(trainset)):
        vid = trainset[i]['id']
        cap = trainset[i]['caption']
        for j in range(len(cap)):
            cap[j] = '<BOS> '+cap[j]+' <EOS>'
        Vid2Cap_train[vid] = cap
        
    Vid2Cap_test = {}
    for i in range(len(testset)):
        vid = testset[i]['id']
        cap = testset[i]['caption']
        for j in range(len(cap)):
            cap[j] = '<BOS> '+cap[j]+' <EOS>'
        Vid2Cap_test[vid] = cap
        
    word_counts, unk_required = build_vocab(trainset,testset,word_count_threshold=0)
    word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)


    # Set hyper params
    EPOCH = 30
    BATCH_SIZE = 10
    nIter = 2000
    LEARNING_RATE = 0.01 
    vovab_size = len(word_counts)
    import_model = False
    show_image_wall = False
    
    # initial model
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    print(s2vt)
    #exit(0)

    if import_model:
        s2vt.load_state_dict(torch.load(model_path+model_file))
    
    s2vt = s2vt.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt.parameters(), lr=LEARNING_RATE)
    loss_list = []

    # training start
    for epoch in range(EPOCH):
        for i in range(nIter):
            video, caption, cap_mask, captions_list, vid = get_train_data(BATCH_SIZE)
            video, caption, cap_mask = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), torch.FloatTensor(cap_mask).cuda()

            cap_out = s2vt(video, caption)
            cap_labels = caption[:, 1:].contiguous().view(-1)      
            cap_mask = cap_mask[:, 1:].contiguous().view(-1)       

            logit_loss = loss_func(cap_out, cap_labels)
            masked_loss = logit_loss*cap_mask
            loss = torch.sum(masked_loss)/torch.sum(cap_mask)
            loss_list.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%nIter == 0:
                time_end = time.time()              
                time_cost = time_end-time_start
                print("Epoch: %d,  loss: %f, time: %f" % (epoch,loss, time_cost))
                time_start = time.time()
                #write_txt(epoch, i, loss, time_cost)
                
            if i%nIter == 0:
                torch.save(s2vt.state_dict(), model_path+model_file)
                print("Epoch: %d value saved" % (epoch))

    time_end_all = time.time()
    print("Training Completed")
    
    # save test output to text
    BATCH_SIZE = 1

    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    
    s2vt.load_state_dict(torch.load(model_path+model_file))
    s2vt.eval()

    for i in range(len(video_test)):
        video_name = video_test[i]
        video = np.array(np.load(VIDEO_DIR_TEST+video_name+'.npy'))
        video = torch.FloatTensor(video).cuda()
        cap_out = s2vt(video)
        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())
        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]
        output = print_in_english(captions)
        output_txt(output_filename, video_name, output)
    print("Testing completed! Output text file saved!")
       

                
                






