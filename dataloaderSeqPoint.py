import random
import numpy as np
import torch
import torch.utils.data as data_utils

torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloaderSeqPoint():
    def __init__(self, args, dataset, bert=False):
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size

        if bert:
            self.train_dataset, self.val_dataset = dataset.get_bert_datasets(mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset = dataset.get_datasetsSeqPoint()

    @classmethod
    def code(cls):
        return 'dataloaderSeqPoint'

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


class NILMDatasetToPoint(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=1):
        """
        初始化NILMDataset类
        :param x: 输入数据，形状为 (n_samples, n_features)
        :param y: 输出数据，形状为 (n_samples,)
        :param status: 状态数据，形状为 (n_samples,)
        :param window_size: 输入窗口的大小
        :param stride: 滑动窗口的步长
        """
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride
        # 直接设置避免其他问题
        # self.stride = 1

    def __len__(self):
        """
        计算数据集的长度
        """
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    # def __getitem__(self, index):
    #     """
    #     获取单个样本
    #     :param index: 样本索引
    #     :return: 输入窗口和对应的单点输出
    #     """
    #     start_index = index * self.stride
    #     end_index = start_index + self.window_size
    #
    #     # 获取输入窗口
    #     x_window = self.x[start_index:end_index]
    #
    #     # 获取目标点（窗口的最后一个点）
    #     y_target = self.y[end_index - 1]
    #     status_target = self.status[end_index - 1]
    #
    #     # 如果输入窗口不足window_size，进行填充
    #     if len(x_window) < self.window_size:
    #         x_window = self.padding_seq(x_window)
    #
    #     # return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32), torch.tensor(status_target, dtype=torch.float32)
    #     return torch.tensor(x_window), torch.tensor(y_target), torch.tensor(status_target)

    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        :return: 输入窗口和对应的中间点输出
        """
        start_index = index * self.stride
        end_index = start_index + self.window_size

        # 获取输入窗口
        x_window = self.x[start_index:end_index]

        # 计算中间点索引
        mid_index = start_index + self.window_size // 2

        # 获取目标点（窗口的中间点）
        y_target = self.y[mid_index]
        status_target = self.status[mid_index]

        # 如果输入窗口不足window_size，进行填充
        if len(x_window) < self.window_size:
            x_window = self.padding_seq(x_window)

        # 返回输入窗口和对应的中间点目标值
        return torch.tensor(x_window), torch.tensor(y_target), torch.tensor(status_target)

    def padding_seq(self, in_array):
        """
        对不足window_size的序列进行填充
        :param in_array: 输入序列
        :return: 填充后的序列
        """
        if len(in_array) == self.window_size:
            return in_array

        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except IndexError:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array





