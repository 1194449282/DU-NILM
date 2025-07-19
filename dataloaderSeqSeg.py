import random
import numpy as np
import torch
import torch.utils.data as data_utils


torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloaderSeqSeg():
    def __init__(self, args, dataset, bert=False):
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size

        if bert:
            self.train_dataset, self.val_dataset = dataset.get_bert_datasets(mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset = dataset.get_datasetsSeqSubSeqSCS()

    @classmethod
    def code(cls):
        return 'NILMDataloaderSeqSeg'

    def get_dataloaders(self):
        train_loader = self._get_loader_train(self.train_dataset)
        val_loader = self._get_loader(self.val_dataset)
        return train_loader, val_loader

    def get_dataloaders_test(self):
        val_loader = self._get_loader(self.val_dataset)
        return val_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_loader_train(self, dataset):
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        return dataloader

class NILMDatasetToSubseqSCS(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=1):
        """
        初始化NILMDataset类
        :param x: 输入数据，形状为 (n_samples, n_features)
        :param y: 输出数据，形状为 (n_samples,)
        :param status: 状态数据，形状为 (n_samples,)
        :param window_size: 输入窗口的大小
        :param subseq_size: 输出子序列的大小
        :param stride: 滑动窗口的步长
        """
        # 1 需要确保 子序列 可以整除 连续点  如60//20 = 3
        # 2 确保 其余能够整除 如600-60=540   540//3+1= 前后隔开135

        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.subseq_size = 240
        self.stride = stride
        self.continuous_length = 80

    def __len__(self):
        """
        计算数据集的长度
        """
        # 确保输入窗口和输出子序列的长度足够
        return int(np.ceil((len(self.x) - self.window_size - self.subseq_size + 1) / self.stride))


    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        :return: 输入窗口和对应的输出子序列
        """
        start_index = index * self.stride
        end_index = start_index + self.window_size

        # 获取输入窗口
        x_window = self.x[start_index:end_index]

        # 计算非输出间隔点
        if self.subseq_size > self.continuous_length:
            non_output_interval = (self.window_size - self.subseq_size) // (
                        self.subseq_size // self.continuous_length + 1)   #15
        else:
            non_output_interval = 0  # 如果连续子序列长度等于子序列总长度，则没有间隔

        # 获取输出子序列
        y_subseq = []
        status_subseq = []
        for i in range(self.subseq_size // self.continuous_length):
            start = start_index + (non_output_interval*(i+1)) + i * self.continuous_length
            end = start + self.continuous_length
            y_subseq.append(self.y[start:end])
            status_subseq.append(self.status[start:end])

        # 将列表展平为一维序列
        y_subseq = [item for sublist in y_subseq for item in sublist]
        status_subseq = [item for sublist in status_subseq for item in sublist]

        y_subseq = np.array(y_subseq)
        status_subseq = np.array(status_subseq)

        # 如果输入窗口不足window_size，进行填充
        if len(x_window) < self.window_size:
            x_window = self.padding_seq(x_window, self.window_size)

        # 如果输出子序列不足subseq_size，进行填充
        expected_length = self.subseq_size
        if len(y_subseq) < expected_length:
            y_subseq = self.padding_seq(y_subseq, expected_length)
        if len(status_subseq) < expected_length:
            status_subseq = self.padding_seq(status_subseq, expected_length)

        return (
            torch.tensor(x_window),
            torch.tensor(y_subseq),
            torch.tensor(status_subseq)
        )
    def padding_seq(self, in_array, target_length):
        """
        对不足指定长度的序列进行填充
        :param in_array: 输入序列
        :param target_length: 目标长度
        :return: 填充后的序列
        """
        if len(in_array) == target_length:
            return in_array

        try:
            if len(in_array.shape) > 1:
                out_array = np.zeros((target_length, in_array.shape[1]))
            else:
                out_array = np.zeros(target_length)
        except AttributeError:
            out_array = np.zeros(target_length)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array

