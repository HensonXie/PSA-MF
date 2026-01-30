import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 词表构建
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

def return_unk():
    return UNK

def get_length(x):
    return x.shape[1]-(np.sum(x, axis=-1) == 0).sum(1)

class MOSI:
    def __init__(self, config):
        if config.sdk_dir:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        
        try:
            self.train = load_pickle(os.path.join(DATA_PATH, 'train.pkl'))
            self.dev = load_pickle(os.path.join(DATA_PATH, 'dev.pkl'))
            self.test = load_pickle(os.path.join(DATA_PATH, 'test.pkl'))
            self.pretrained_emb, self.word2id = None, None
            print(f"Loaded processed data from {DATA_PATH}")

        except Exception as e:
            print(f"Could not load processed data: {e}")
            print(f"Attempting to process raw data in {DATA_PATH}...")

            
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH, exist_ok=True)

            pickle_filename = os.path.join(DATA_PATH, 'mosi_data_noalign.pkl')
            csv_filename = os.path.join(DATA_PATH, 'MOSI-label.csv')

            if not os.path.exists(pickle_filename) or not os.path.exists(csv_filename):
                raise FileNotFoundError(
                    f"Raw data files not found at {DATA_PATH}. "
                    "Please ensure 'mosi_data_noalign.pkl' and 'MOSI-label.csv' are present, "
                    "or provide processed 'train.pkl/dev.pkl/test.pkl'."
                )

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']
            
            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            pattern = re.compile('(.*)_(.*)')
            
            # 数据拼接与处理逻辑
            v = np.concatenate((train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)

            a = np.concatenate((train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
            alens = get_length(a)
            
            label = np.concatenate((train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
            
            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            all_id_list = list(map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                idd1, idd2 = re.search(pattern, idd).group(1, 2)
                try:
                    index = all_csv_id.index((idd1, idd2))
                except ValueError:
                    # 如果找不到对应的 CSV 条目，跳过
                    continue

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]

                # remove nan
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                actual_words = list(_words)
                words = [] 

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                data_sample = ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd)

                if i < dev_start:
                    train.append(data_sample)
                elif i >= dev_start and i < test_start:
                    dev.append(data_sample)
                elif i >= test_start:
                    test.append(data_sample)

            print(f"Processed Raw Data. Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
            word2id.default_factory = return_unk

            # 保存处理后的数据
            to_pickle(train, os.path.join(DATA_PATH, 'train.pkl'))
            to_pickle(dev, os.path.join(DATA_PATH, 'dev.pkl'))
            to_pickle(test, os.path.join(DATA_PATH, 'test.pkl'))

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class MOSEI:
    def __init__(self, config):
        if config.sdk_dir:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        
        try:
            self.train = load_pickle(os.path.join(DATA_PATH, 'train.pkl')) 
            self.dev = load_pickle(os.path.join(DATA_PATH, 'dev.pkl'))
            self.test = load_pickle(os.path.join(DATA_PATH, 'test.pkl'))
            self.pretrained_emb, self.word2id = None, None
            print(f"Loaded processed MOSEI data from {DATA_PATH}")

        except Exception as e:
            print(f"Could not load processed MOSEI data: {e}. Attempting to process raw data...")
            
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH, exist_ok=True)

            pickle_filename = os.path.join(DATA_PATH, 'mosei_senti_data_noalign.pkl')
            csv_filename = os.path.join(DATA_PATH, 'MOSEI-label.csv')
            
            if not os.path.exists(pickle_filename) or not os.path.exists(csv_filename):
                 raise FileNotFoundError(f"Raw MOSEI files not found in {DATA_PATH}")

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            pattern = re.compile('(.*)_([.*])')

            v = np.concatenate((train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)
            a = np.concatenate((train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
            alens = get_length(a)
            label = np.concatenate((train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])
            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                index = i 
                
                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                actual_words = list(_words)
                words = []
                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                data_sample = ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd)

                if i < dev_start:
                    train.append(data_sample)
                elif i >= dev_start and i < test_start:
                    dev.append(data_sample)
                elif i >= test_start:
                    test.append(data_sample)

            print(f"Processed MOSEI. Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
            word2id.default_factory = return_unk
            self.pretrained_emb = None

            to_pickle(train, os.path.join(DATA_PATH, 'train.pkl'))
            to_pickle(dev, os.path.join(DATA_PATH, 'dev.pkl'))
            to_pickle(test, os.path.join(DATA_PATH, 'test.pkl'))

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()