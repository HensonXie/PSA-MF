import torch
import argparse
import numpy as np
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader

# 通用设置
def set_seed(seed):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_dtype(torch.float32) 
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')过时写法，不建议使用，容易出错 
        #“Explicit is better than implicit”（显式优于隐式）,显式地使用 model.to(device) 和 data.to(device)，而不是通过修改全局默认 Tensor 类型来隐式实现，那样会让代码非常难以调试
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(args.seed)
    
    print(f"Loading data for {dataset}...")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size, sdk_path=args.sdk_path)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, sdk_path=args.sdk_path)
    test_config = get_config(dataset, mode='test', batch_size=args.batch_size, sdk_path=args.sdk_path)

    # 这里的 train_loader 会使用 args.bert_path 加载模型
    train_loader = get_loader(args, train_config, shuffle=True)
    valid_loader = get_loader(args, valid_config, shuffle=False)
    test_loader = get_loader(args, test_config, shuffle=False)
    
    print('Data loaded successfully.')

    # 注入 word2id 到 args (如果模型需要)
    args.word2id = train_config.word2id
    
    # 设置模型维度参数
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = dataset
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True, device=device)
    solver.train_and_eval()