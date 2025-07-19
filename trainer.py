import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

import os
import json
import random
import numpy as np
from abc import *
from pathlib import Path

from utils import *
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.DoubleTensor)


class Trainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, stats, export_root):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.export_root = Path(export_root)

        self.cutoff = torch.tensor([args.cutoff[i]
                                    for i in args.appliance_names]).to(self.device)
        self.threshold = torch.tensor(
            [args.threshold[i] for i in args.appliance_names]).to(self.device)
        self.min_on = torch.tensor([args.min_on[i]
                                    for i in args.appliance_names]).to(self.device)
        self.min_off = torch.tensor(
            [args.min_off[i] for i in args.appliance_names]).to(self.device)

        self.normalize = args.normalize
        self.denom = args.denom
        if self.normalize == 'mean':
            self.x_mean, self.x_std = stats
            self.x_mean = torch.tensor(self.x_mean).to(self.device)
            self.x_std = torch.tensor(self.x_std).to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).to(self.device)
        print('C0: {}'.format(self.C0))
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss(reduction='mean')
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='mean')

    def train(self, isFirst):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []
        if isFirst:
            best_rel_err = np.array(999.0)
            best_mae_err = np.array(999.0)
            best_mse = np.array(99999.0)
            best_acc = np.array(0.0)
            best_f1 = np.array(0.0)
        else:
            best_rel_err, best_mae_err, best_acc, _, _, best_f1, best_mse = self.validate()
            self._save_state_dict()

        early_stopping_patience = 15  # 设置早停的耐心值
        early_stopping_counter = 0  # 初始化早停计数器
        for epoch in range(self.num_epochs):
            self.train_bert_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1, mse = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())
            print(f"F1 {f1.mean()} ")
            print(f"MAE {abs_err.mean()} ")
            print(f"MSE {mse.mean()} ")
            # if f1.mean() + acc.mean() - abs_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
            # if f1.mean() + acc.mean() - abs_err.mean() > best_f1.mean() + best_acc.mean() - best_mae_err.mean():
            if f1.mean() + acc.mean() - abs_err.mean() > best_f1.mean() + best_acc.mean() - best_mae_err.mean():
            # if f1.mean()  - abs_err.mean() > best_f1.mean() + best_acc.mean() - best_mae_err.mean():
            # if f1.mean()  - mse.mean() > best_f1.mean()- best_mse.mean():
            # if f1.mean() - rel_err.mean() > best_f1.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                best_mae_err = abs_err
                best_mse = mse
                self._save_state_dict()
                early_stopping_counter = 0  # 重置早停计数器
            else:
                early_stopping_counter += 1  # 早停计数器加1
            print(f"早停 {early_stopping_counter} epochs.")
        # 判断是否触发早停机制
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


    def train_bert_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        loss_values, relative_errors, absolute_errors = [], [], []
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
            batch_shape = status.shape
            self.optimizer.zero_grad()
            logits = self.model(seqs) #输出
            labels = labels_energy / self.cutoff
            logits_energy = self.cutoff_energy(logits * self.cutoff) # 去掉小于5的数据，最大取cutoff（# 限制上限）里的数据
            logits_status = self.compute_status(logits_energy) # 获取状态


            mask = (status >= 0)
            labels_masked = torch.masked_select(labels, mask).view((-1, batch_shape[-1]))
            logits_masked = torch.masked_select(logits, mask).view((-1, batch_shape[-1]))
            status_masked = torch.masked_select(status, mask).view((-1, batch_shape[-1]))
            logits_status_masked = torch.masked_select(logits_status, mask).view((-1, batch_shape[-1]))

            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).double(),
                labels_masked.contiguous().view(-1).double())
            # mae_loss = self.l1_on(logits_masked.contiguous().view(-1).double(),
            #     labels_masked.contiguous().view(-1).double())
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).double(),
                (status_masked * 2 - 1).contiguous().view(-1).double())
            total_loss = kl_loss + mse_loss + margin_loss
            # total_loss =  mse_loss + margin_loss
            # total_loss = mse_loss*0.4+mae_loss*0.4+margin_loss


            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                    labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

    def validate(self):
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []
        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch #seqs 1 480 =batchsize,window_size
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
                logits = self.model(seqs)
                # labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff) # 结果 * cutoff  小于5则0 最大不超过cutoff
                logits_status = self.compute_status(logits_energy) # 判断状态开启还是关闭
                # logits_energy = logits_energy * logits_status # 去掉关闭时功率为0

                rel_err, abs_err,a = relative_absolute_error(logits_energy.detach(
                ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())



                average_acc = np.mean(np.concatenate(acc_values))
                # average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.concatenate(f1_values))
                # average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.concatenate(relative_errors))
                # average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                label_curve.append(labels_energy.detach().cpu().numpy().tolist())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().tolist())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())
                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))


        label_curve = np.concatenate(label_curve)
        e_pred_curve = np.concatenate(e_pred_curve)
        status_curve = np.concatenate(status_curve)
        s_pred_curve = np.concatenate(s_pred_curve)
        return_rel_err1, return_abs_err1, return_acc1, return_precision1, return_recall1, return_f11 = (
            np.mean(np.concatenate(arr), axis=0) for arr in
            [relative_errors, absolute_errors, acc_values, precision_values, recall_values, f1_values]
        )
        return_rel_err, return_abs_err,mse = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score(s_pred_curve,
                                                                                               status_curve)

        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1,mse

    def test(self, test_loader):
        self._load_best_model()
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []
        acc_values1, precision_values1, recall_values1, f1_values1,  = [], [], [], []
        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch  # 这里的labels_energy已经被限制cutoff
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
                logits = self.model(seqs)

                seqs = seqs * self.x_std + self.x_mean

                logits_energy = self.cutoff_energy(logits * self.cutoff)    # 估计值*上限 然后去掉<5的值
                logits_status = self.compute_status(logits_energy)  # 根据值去评估 若>threshold 则开启

                # 计算这些需要估计出状态  logits_status
                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                    ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())

                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                rel_err, abs_err,mse = relative_absolute_error(logits_energy.detach(
                    ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                average_acc = np.mean(np.concatenate(acc_values))
                average_f1 = np.mean(np.concatenate(f1_values))
                average_rel_err = np.mean(np.concatenate(relative_errors))

                acc1, precision1, recall1, f11 = acc_precision_recall_f1_score1111(logits_status.detach(
                ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values1.append(acc1.tolist())

                f1_values1.append(f11.tolist())
                average_acc1 = np.mean(acc_values1)
                average_f11 = np.mean(f1_values1)


                tqdm_dataloader.set_description('Test, rel_err {:.5f}, acc {:.5f}, f1 {:.5f}, accm {:.5f}, f1m {:.5f}'.format(
                    average_rel_err, average_acc, average_f1,average_acc1,average_f11))

                label_curve.append(labels_energy.detach().cpu().numpy().tolist())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().tolist())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())

        label_curve = np.concatenate(label_curve)
        e_pred_curve = np.concatenate(e_pred_curve)
        status_curve = np.concatenate(status_curve)
        s_pred_curve = np.concatenate(s_pred_curve)

        self._save_result({'gt': label_curve.tolist(),
            'pred': e_pred_curve.tolist()}, 'test_result.json')


        return_rel_err, return_abs_err,mse = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score(s_pred_curve, status_curve)

        SAE = symmetric_absolute_error(e_pred_curve, label_curve)
        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1,SAE

    def cutoff_energy(self, data): # 此方法 去掉小于5的数据，最大取cutoff（# 限制上限）里的数据

        try:
            columns = data.squeeze().shape[-1]
        except IndexError:
            columns = 1

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).to(self.device)

        data[data < 5] = 0
        data = torch.min(data, self.cutoff.double())
        return data

    def compute_status(self, data):
        data_shape = data.shape
        # columns = data.squeeze().shape[-1]
        try:
            columns = data.squeeze().shape[-1]
        except IndexError:
            columns = 1
        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).to(self.device)
        
        status = (data >= self.threshold) * 1
        return status

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)
