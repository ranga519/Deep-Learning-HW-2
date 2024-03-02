import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

test = json.load(open('/home/ranga519/Deep_Learning_HW2/hw2_4/testing_label.json'))
output = '/home/ranga519/Deep_Learning_HW2/hw2_4/test_output_edit.txt'
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))
