import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def get_user_input(args):
    if torch.cuda.is_available():
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    dataset_code = {'r': 'redd_lf', 'u': 'uk_dale'}
    args.dataset_code = dataset_code[input(
        'Input r for REDD, u for UK_DALE: ')]

    if args.dataset_code == 'redd_lf':
        app_dict = {
            'r': ['refrigerator'],
            'w': ['washer_dryer'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input r, w, m or d for target appliance: ')]

    elif args.dataset_code == 'uk_dale':
        app_dict = {
            'k': ['kettle'],
            'f': ['fridge'],
            'w': ['washing_machine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input k, f, w, m or d for target appliance: ')]

    args.num_epochs = int(input('Input training epochs: '))
    model_dict = {
        '1': 'BERT4NILM',
        '2': 'seq2ponitcnn_Pytorch',
        '3': 'seq2seqcnn_Pytorch',
        '4': 'seq2Subcnn_Pytorch',
        '5': 'UNETNiLM',
        '6': 'TCN',
        '7': 'ELECTRICITY',
        '8': 'DU-NILM',
    }
    args.model = model_dict[input(
        'Input model: ')]

def set_template(args):
    args.output_size = len(args.appliance_names)
    if args.dataset_code == 'redd_lf':

        # 限制上限
        args.cutoff = {
            'aggregate': 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave': 1800,
            'dishwasher': 1200
        }

        #高于此值则为开启状态
        args.threshold = {
            'refrigerator': 50,
            'washer_dryer': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        # 设备开启持续时长必须大于
        args.min_on = {
            'refrigerator': 10,
            'washer_dryer': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        # 设备关闭-开启的间隔大于此
        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'refrigerator': 1e-6,
            'washer_dryer': 0.001,
            'microwave': 1.,
            'dishwasher': 1.
        }


    elif args.dataset_code == 'uk_dale':

        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.
        }

    args.enable_lr_schedule = False


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def acc_precision_recall_f1_score1111(pred, status):
    assert pred.shape == status.shape, "Predictions and status must have the same shape"

    # 将预测值和真实标签展平为一维数组
    pred_flat = pred.flatten()
    status_flat = status.flatten()

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(status_flat, pred_flat, labels=[0, 1]).ravel()

    # 计算指标
    acc = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / np.max((tp + fp, 1e-9))
    recall = tp / np.max((tp + fn, 1e-9))
    f1_score = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

    return acc, precision, recall, f1_score

def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, mse = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))
        mse_error = np.mean((label[:, i] - pred[:, i]) ** 2)

        relative.append(relative_error)
        absolute.append(absolute_error)
        mse.append(mse_error)

    return np.array(relative), np.array(absolute), np.array(mse)


def symmetric_absolute_error(pred, label):
    assert pred.shape == label.shape

    # 确保预测和标签的形状是正确的
    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])

    # 初始化 SAE 数组
    sae = []

    # 计算每个标签的 SAE
    for i in range(label.shape[-1]):
        total_pred = np.sum(pred[:, i])
        total_label = np.sum(label[:, i])
        sae.append(np.abs(total_pred - total_label) / total_label if total_label != 0 else 0)

    return np.array(sae)



