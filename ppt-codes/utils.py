import os
import csv
import random
import requests
import io
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from data import Dictionary

def build_words_dict(load_raw_file,save_dict_file):
    word_dict = Dictionary()
    with open(load_raw_file,mode="r",encoding="gbk") as rfp:
        for line in rfp:
            sep_words = line.strip().split()
            for word in sep_words:
                word_dict.add(word)
    word_dict.save(save_dict_file)
def download(raw_url,save_file_name):
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
    }
    r=requests.get(raw_url,headers=headers)
    with open(save_file_name,'wb') as f:
        f.write(r.content)
def get_images(load_csv_file,save_data_path,num_pics = 4000):
    images_path = os.path.join(save_data_path,"images")
    save_csv_file = os.path.join(save_data_path,"id_text.csv")
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    with open(load_csv_file,mode='r',encoding="utf-8") as rfp:
        reader = csv.reader(rfp)
        next(reader)
        idx = 0
        with open(save_csv_file,mode="w",encoding="utf-8") as rfp_w:
            csv_writer = csv.writer(rfp_w)
            csv_writer.writerow(["index","text"])
            for row in tqdm(reader,desc="pic process"):
                #the first row is table header
                pic_url = row[0]
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                }
                r=requests.get(pic_url,headers=headers)
                idx += 1
                if r.status_code == 200:
                    raw_img = Image.open(io.BytesIO(r.content))
                    img = np.array(raw_img)
                    if len(img.shape)<3:
                        continue
                    save_file_name = os.path.join(images_path,"img-%d.jpg"%idx)
                    with open(save_file_name,'wb') as f:
                        f.write(r.content)
                    raw_text = row[1]
                    csv_writer.writerow([idx,raw_text])
                else:
                    continue
                if idx>num_pics:
                    break
        
def split_dataset(load_path,save_path,train=0.70,valid=0.15):
    imgs_path = os.path.join(load_path,"images")
    all_rows =list(range(len(os.listdir(imgs_path))))
    train_len,valid_len = int(len(all_rows)*train),int(len(all_rows)*(train+valid))
    all_index_rows = list(range(len(all_rows)))
    random.shuffle(all_index_rows)
    train_rows = all_index_rows[:train_len]
    valid_rows = all_index_rows[train_len:valid_len]
    test_rows = all_index_rows[valid_len:]
    train_file = os.path.join(save_path,"train.idx")
    valid_file = os.path.join(save_path,"valid.idx")
    test_file = os.path.join(save_path,"test.idx")
    with open(train_file,mode="w",encoding="utf-8") as rfp:rfp.write("\t".join(map(str,train_rows)))
    with open(valid_file,mode="w",encoding="utf-8") as rfp:rfp.write("\t".join(map(str,valid_rows)))
    with open(test_file,mode="w",encoding="utf-8") as rfp:rfp.write("\t".join(map(str,test_rows)))


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)