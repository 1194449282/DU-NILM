import random
import numpy as np
import torch
import torch.utils.data as data_utils


torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloaderSeqSub():
    def __init__(self, args, dataset, bert=False):
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size

        if bert:
            self.train_dataset, self.val_dataset = dataset.get_bert_datasets(mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset = dataset.get_datasetsSeqSubSeq()

    @classmethod
    def code(cls):
        return 'NILMDataloaderSeqSub'

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

class NILMDatasetToSubseq(data_utils.Dataset):
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
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.subseq_size = 120
        self.stride = stride

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
        # end_index = start_index + self.window_size + self.subseq_size

        # 获取输入窗口
        x_window = self.x[start_index:start_index + self.window_size]

        # 计算中间点的索引
        middle_index = start_index + (self.window_size-self.subseq_size) // 2

        # 获取输出子序列
        y_subseq = self.y[middle_index:middle_index + self.subseq_size]
        status_subseq = self.status[middle_index:middle_index + self.subseq_size]

        # 如果输入窗口不足window_size，进行填充
        if len(x_window) < self.window_size:
            x_window = self.padding_seq(x_window)

        # 如果输出子序列不足subseq_size，进行填充
        if len(y_subseq) < self.subseq_size:
            y_subseq = self.padding_seq(y_subseq)
        if len(status_subseq) < self.subseq_size:
            status_subseq = self.padding_seq(status_subseq)

        return (
            torch.tensor(x_window),
            torch.tensor(y_subseq),
            torch.tensor(status_subseq)
        )

    def padding_seq(self, in_array):
        """
        对不足指定长度的序列进行填充
        :param in_array: 输入序列
        :return: 填充后的序列
        """
        if len(in_array) == self.window_size or len(in_array) == self.subseq_size:
            return in_array

        try:
            if len(in_array.shape) > 1:
                out_array = np.zeros((self.window_size, in_array.shape[1]))
            else:
                out_array = np.zeros(self.window_size)
        except AttributeError:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array


