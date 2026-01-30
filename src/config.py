import argparse
import os
import pprint
from pathlib import Path
import torch.optim as optim
import torch.nn as nn

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {
    'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
    "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, 
    "rrelu": nn.RReLU, "tanh": nn.Tanh
}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    
    # Paths
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='dataset to use')
    parser.add_argument('--data_path', type=str, default=str(data_dir),
                        help='path for storing the dataset')
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased',
                        help='path or name of pretrained BERT model')
    parser.add_argument('--personality_bert_path', type=str, default='bert-base-uncased',
                        help='path or name of pretrained Personality BERT model')

    parser.add_argument('--sdk_path', type=str, default=None,
                        help='path to CMU-MultimodalSDK (optional if pickles exist)')
    
    # Tasks
    parser.add_argument('--v_only', action='store_true', help='use visual modality only')
    parser.add_argument('--a_only', action='store_true', help='use acoustic modality only')
    parser.add_argument('--l_only', action='store_true', help='use language modality only')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1, help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1, help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1, help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--n_layer', type=int, default=1, help='number of layers in LSTM encoders')
    parser.add_argument('--d_vh', type=int, default=64, help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=64, help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=64, help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=64, help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128, help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768, help='dimension of pretrained model output')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size') #MOSI 256 MOSEI 128
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--lr_main', type=float, default=5e-4, help='initial learning rate for main model') #MOSI 5e4 MOSEI 1e4
    parser.add_argument('--lr_bert', type=float, default=5e-5, help='initial learning rate for bert') #MOSI 5e5 MOSEI 1e5
    parser.add_argument('--lr_mmilb', type=float, default=1e-4, help='initial learning rate for mmilb') #MOSI 1e4 MOSEI 4e4
    
    parser.add_argument('--weight_decay_main', type=float, default=1e-4, help='L2 penalty factor')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4, help='L2 penalty factor')
    
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--when', type=int, default=20, help='when to decay learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--update_batch', type=int, default=1, help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100, help='frequency of result logging')
    parser.add_argument('--seed', type=int, default=1007, help='random seed')
    parser.add_argument('--name', type=str, default='psa_mf', help='name for saving model')

    # Transformer specific
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--embed_dropout', type=float, default=0.25)
    parser.add_argument('--res_dropout', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--attn_mask', action='store_true', default=True)

    args = parser.parse_args()
    return args

class Config(object):
    def __init__(self, data, mode='train', sdk_path=None):
        self.dataset_dir = data_dir.joinpath(data.upper())
        self.sdk_dir = sdk_path 
        self.mode = mode

    def __str__(self):
        return pprint.pformat(self.__dict__)

def get_config(dataset='mosi', mode='train', batch_size=32, sdk_path=None):
    config = Config(data=dataset, mode=mode, sdk_path=sdk_path)
    config.dataset = dataset
    config.batch_size = batch_size
    return config