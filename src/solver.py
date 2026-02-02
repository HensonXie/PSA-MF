import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
# 引入评估函数
from utils.eval_metrics import eval_mosi, eval_mosei_senti
from utils.tools import save_model
from model import PSAMF
from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, device=None):
        self.hp = hyp_params
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.is_train = is_train
        self.update_batch = hyp_params.update_batch
        self.model = model if model else PSAMF(hyp_params).to(self.device)

        self.criterion = nn.CrossEntropyLoss() if self.hp.dataset == "ur_funny" else nn.L1Loss()

        if self.is_train:
            self.init_optimizer()
            self.scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_main, mode='min', patience=hyp_params.when, factor=0.5, verbose=True
            )

        log_path = os.path.join(os.getcwd(), 'logs', self.hp.name)
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path)

    def init_optimizer(self):
        bert_param, mmilb_param, main_param = [], [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    bert_param.append(param)
                elif 'mi' in name:
                    mmilb_param.append(param)
                else:
                    main_param.append(param)
        
        for param in mmilb_param + main_param:
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

        optimizer_grouped_parameters = [
            {'params': bert_param, 'weight_decay': self.hp.weight_decay_bert, 'lr': self.hp.lr_bert},
            {'params': main_param, 'weight_decay': self.hp.weight_decay_main, 'lr': self.hp.lr_main}
        ]
        self.optimizer_main = getattr(optim, self.hp.optim)(optimizer_grouped_parameters)

    def train_and_eval(self):
        best_valid_loss = float('inf')
        best_mae = float('inf')
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs + 1):
            start = time.time()
            
            # 1. Train
            train_loss = self.train(epoch)
            
            # 2. Validate
            val_loss, val_results, val_truths = self.evaluate(test=False)
            
            # 3. Test
            test_loss, test_results, test_truths = self.evaluate(test=True)
            
            duration = time.time() - start
            
            if self.is_train:
                self.scheduler_main.step(val_loss)

            print("-" * 50)
            print(f'Epoch {epoch:2d} | Time {duration:5.4f}s | Train Loss {train_loss:.4f} | Valid Loss {val_loss:.4f} | Test Loss {test_loss:.4f}')

            if self.hp.dataset in ["mosi", "mosei_senti", "mosei"]:

                eval_func = eval_mosei_senti if "mosei" in self.hp.dataset else eval_mosi
                
                val_metrics = eval_func(val_results, val_truths, exclude_zero=False)
                test_metrics = eval_func(test_results, test_truths, exclude_zero=False)

                # 格式化输出
                print(f"Valid Metrics: " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
                print(f"Test Metrics : " + ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))

                # TensorBoard Logging
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Loss/test', test_loss, epoch)
                
                # 记录详细指标到 TensorBoard
                for k, v in test_metrics.items():
                    self.writer.add_scalar(f'Test_Metrics/{k}', v, epoch)
            else:
                # 如果是其他数据集，暂时只用 loss
                test_metrics = {'MAE': test_loss}

            # --------------------------------------------------------
            # 保存最佳模型逻辑
            # --------------------------------------------------------
            # 通常主要关注 MAE (对于回归任务)
            current_mae = test_metrics['MAE'] if 'MAE' in test_metrics else test_loss
            
            if val_loss < best_valid_loss:
                patience = self.hp.patience
                best_valid_loss = val_loss
                
                # 如果当前模型的测试集 MAE 也是最好的，保存
                if current_mae < best_mae:
                    best_mae = current_mae
                    print(f"Found new best model (MAE: {best_mae:.4f})! Saving...")
                    save_model(self.hp, self.model, name=self.hp.name)
            else:
                patience -= 1
                if 0:
                    print("Early stopping triggered.")
                    break
            print("-" * 50)
        
        self.writer.close()

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        left_batch = self.update_batch

        for i_batch, batch_data in enumerate(self.train_loader):
            # Unpack data
            text, visual, vlens, audio, alens, y, _, bert_sent, bert_sent_type, bert_sent_mask, _, personality_bert_feature = batch_data
            p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask = personality_bert_feature

            device = self.device
            text, visual, audio, y = text.to(device), visual.to(device), audio.to(device), y.to(device)
            bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
            p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask = p_bert_sentences.to(device), p_bert_sentence_types.to(device), p_bert_sentence_att_mask.to(device)

            if self.hp.dataset == "ur_funny":
                y = y.squeeze()
            
            batch_size = y.size(0)

            # Forward
            nce_list, preds, nce2_list, align_loss = self.model(
                text, visual, audio, vlens, alens, 
                bert_sent, bert_sent_type, bert_sent_mask, y, 
                p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask
            )

            # Loss calc
            y_loss = self.criterion(preds, y)
            loss = y_loss + (nce_list[10] + 0.1 * nce2_list[10]) + 0.09 * align_loss
            loss.backward()

            left_batch -= 1
            if left_batch == 0:
                left_batch = self.update_batch
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.clip)
                self.optimizer_main.step()
                self.model.zero_grad()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size

            if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                print(f'Epoch {epoch:2d} | Batch {i_batch:3d} | Train Loss {avg_loss:5.4f}')
                proc_loss, proc_size = 0, 0

        return epoch_loss / self.hp.n_train

    def evaluate(self, test=False):
        self.model.eval()
        loader = self.test_loader if test else self.dev_loader
        total_loss = 0
        results = []
        truths = []

        with torch.no_grad():
            for batch in loader:
                text, visual, vlens, audio, alens, y, _, bert_sent, bert_sent_type, bert_sent_mask, _, personality_bert_feature = batch
                p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask = personality_bert_feature

                device = self.device
                text, visual, audio, y = text.to(device), visual.to(device), audio.to(device), y.to(device)
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
                p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask = p_bert_sentences.to(device), p_bert_sentence_types.to(device), p_bert_sentence_att_mask.to(device)

                if self.hp.dataset == 'ur_funny':
                    y = y.squeeze()

                _, preds, _, _ = self.model(
                    text, visual, audio, vlens, alens, 
                    bert_sent, bert_sent_type, bert_sent_mask, y, 
                    p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask
                )
                
                loss = self.criterion(preds, y)
                total_loss += loss.item() * y.size(0)
                
                # 收集预测值和真实值用于后续计算指标
                # 注意：这里我们把 tensor 转为 CPU 上的 numpy 或 list，防止显存爆炸
                results.append(preds)
                truths.append(y)

        avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
        
        # 将结果拼接在一起
        results = torch.cat(results)
        truths = torch.cat(truths)
        
        # 注意：这里 evaluate 返回 tensor 供 eval_metrics 使用 (它内部处理了 cpu().numpy())
        return avg_loss, results, truths
