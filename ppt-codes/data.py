import json
import csv
import logging
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        self.start_index = 0
        self.end_index = len(self.ind2token)
    @property
    def pad(self):
        return self.ind2token[0]
    @property
    def start(self):
        return self.ind2token[1]
    @property
    def end(self):
        return self.ind2token[2]
    @property
    def unk(self):
        return self.ind2token[3]
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            return self.token2ind.get(item,self.token2ind[self.unk])
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word)
            self.end_index = len(self.ind2token)
    def save(self,save_file):
        with open(save_file,"w",encoding='utf-8') as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind
            }
            json.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding='utf-8') as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))

class Logger:
    def __init__(self, path,clevel = logging.DEBUG,Flevel = logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        #设置文件日志
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)

    def warn(self,message):
        self.logger.warn(message)

    def error(self,message):
        self.logger.error(message)

    def critical(self,message):
        self.logger.critical(message)

            





class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self,data_path,split,word_dict,caption_len = 64,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.caption_len = caption_len
        self.word_dict = word_dict
        assert self.split in {'train', 'valid', 'test'}

        idx_file_name = os.path.join(data_path,"processed","%s.idx"%self.split)
        self.index_list = self.read_idx(idx_file_name)

        content_file_name = os.path.join(data_path,"raw","id_text.csv")
        self.real_ids,self.content_list = self.read_content(content_file_name)
        
        imgs_path = os.path.join(data_path,"raw","images")
        self.imgs_list = self.read_imgs(imgs_path)
        
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.real_ids)
    def read_idx(self,file_name):
        with open(file_name,mode="r",encoding="utf-8") as rfp:
            idx_list = [int(idx) for idx in rfp.read().split()]
            return idx_list
    def read_content(self,file_name):
        with open(file_name,mode="r",encoding="utf-8") as rfp:
            csv_reader = csv.reader(rfp)
            next(csv_reader)
            all_dataset = {row[0]:row[1] for row in csv_reader}
            keys_list = list(all_dataset.keys())
            real_ids = [keys_list[idx] for idx in self.index_list]
            content_list = [all_dataset[idx] for idx in real_ids]
            return real_ids,content_list
    def read_imgs(self,imgs_path):
        imgs_list = []
        for idx in self.real_ids:
            file_name = os.path.join(imgs_path,"img-%s.jpg"%idx)
            img_mat = np.array(Image.open(file_name))
            imgs_list.append(img_mat)
        return imgs_list
    def __getitem__(self,idx):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # img = torch.FloatTensor(self.imgs_list[idx] / 255.)
        img = self.imgs_list[idx].transpose(2,0,1)
        if self.transform is not None:
            img = self.transform(img)
        sentence = [self.word_dict[word] for word in self.content_list[idx]]
        sentence = [self.word_dict[self.word_dict.start]]+sentence+[self.word_dict[self.word_dict.end]]
        
        return self.real_ids[idx],sentence,img
    def __len__(self):
        return self.dataset_size

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def batchfy(batch):
    index_list = [item[0] for item in batch]
    max_len = max([len(item[1]) for item in batch])
    sentence_list = [[item[1][0]] + item[1][1:-1]+[0]*(max_len-len(item[1][1:-1])) + [item[1][-1]] for item in batch]
    
    max_img_width = max([item[2].shape[1] for item in batch ])
    max_img_height = max([item[2].shape[2] for item in batch ])
    imgs_list = [np.pad(item[2],
                        ((0,0),(0,max_img_width-item[2].shape[1]),(0,max_img_height-item[2].shape[2])),'constant')[np.newaxis,:]
                        for item in batch]
    # imgs_list = [item[2] for item in batch]
    lengths = [len(item) for item in sentence_list]
    # img_sizes = [item.shape for item in imgs_list]
    sent_tensor = torch.tensor(sentence_list,dtype=torch.long)
    img_tensor = torch.tensor(np.vstack(imgs_list),dtype=torch.float)
    len_tensor = torch.tensor(np.array(lengths),dtype=torch.long)
    return (index_list,sent_tensor,img_tensor,len_tensor)

