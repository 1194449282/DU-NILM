import argparse

from modelELECTRIcity import ELECTRICITY

from dataset import *
from modelBERT4NILM import BERT4NILM
from modelDUNILM import DU_NILM
from modelSeq2Point import seq2ponitcnn_Pytorch
from modelSeq2Sub import seq2Subcnn_Pytorch
from modelSeq2seq import seq2seqcnn_Pytorch
from modelTCN import TCN
from modelUNet import UNETNiLM
from trainer import *
from utils import *


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(args, export_root=None, resume=True):
    if args.dataset_code == 'redd_lf':
        # args.house_indicies = [2, 3, 4, 5, 6]
        args.house_indicies = [2, 3]
        dataset = REDD_LF_Dataset(args)
    elif args.dataset_code == 'uk_dale':
        # args.house_indicies = [1, 3, 4, 5]
        args.house_indicies = [2]
        dataset = UK_DALE_Dataset(args, IsUK=True)
        args.validation_size = 0.25    # uk数据集按照 2 6 2 划分

    x_mean, x_std = dataset.get_mean_std()
    stats = (x_mean, x_std)
    #  加载模型
    print(args.model)
    dataloader = NILMDataloader(args, dataset, bert=False)
    if args.model == 'BERT4NILM':
        model = BERT4NILM(args)
    elif args.model == 'seq2ponitcnn_Pytorch':
        model = seq2ponitcnn_Pytorch(args.window_size)
        dataloader = NILMDataloaderSeqPoint(args, dataset, bert=False)
    elif args.model == 'seq2seqcnn_Pytorch':
        model = seq2seqcnn_Pytorch(args.window_size)
    elif args.model == 'seq2Subcnn_Pytorch':
        model = seq2Subcnn_Pytorch(args.window_size)
        dataloader = NILMDataloaderSeqSub(args, dataset, bert=False)
        # dataloader = NILMDataloaderSeqSeg(args, dataset, bert=False)
    elif args.model == 'ELECTRICITY':
        model = ELECTRICITY()
    elif args.model == 'TCN':
        model = TCN()
    elif args.model == 'DU-NILM':
        model = DU_NILM()
        dataloader = NILMDataloaderSeqSeg(args, dataset, bert=False)
    elif args.model == 'UNETNiLM':
        model = UNETNiLM(seq_len=args.window_size)

    # 判断是否继续训练
    if export_root == None:
        folder_name = '-'.join(args.appliance_names)
        export_root = 'experiments/' + args.dataset_code + '/' + folder_name + '/' + args.model

    # 加载数据集

    train_loader, val_loader = dataloader.get_dataloaders()

    trainer = Trainer(args, model, train_loader,
                    val_loader, stats, export_root)
    isFirst = True
    if args.num_epochs > 0:
        if resume :
            try:
                model.load_state_dict(torch.load(os.path.join(
                    export_root, 'best_acc_model.pth'), map_location='cpu'))
                print('Successfully loaded previous model, continue training...')
                isFirst = False
            except FileNotFoundError:
                print('Failed to load old model, continue training new model...')
        trainer.train(isFirst)

    args.validation_size = 1.
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [1]
        dataset = REDD_LF_Dataset(args, stats)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [2]
        dataset = UK_DALE_Dataset(args, stats, IsUK=True)

    dataloader = NILMDataloader(args, dataset, bert=False)
    if args.model == 'seq2ponitcnn_Pytorch':
        dataloader = NILMDataloaderSeqPoint(args, dataset, bert=False)
    elif args.model == 'seq2Subcnn_Pytorch':
        dataloader = NILMDataloaderSeqSub(args, dataset, bert=False)
        # dataloader = NILMDataloaderSeqSeg(args, dataset, bert=False)
    elif args.model == 'DU-NILM':
        dataloader = NILMDataloaderSeqSeg(args, dataset, bert=False)
    test_loader = dataloader.get_dataloaders_test()
    rel_err, abs_err, acc, prec, recall, f1, SAE  = trainer.test(test_loader)
    acc_mean = np.mean(acc)
    f1_mean = np.mean(f1)
    rel_err_mean = np.mean(rel_err)
    abs_err_mean = np.mean(abs_err)
    SAE_mean = np.mean(SAE)

    # 打印结果，保留三位小数
    print('Mean Accuracy: {:.3f}'.format(acc_mean))
    print('Mean F1-Score: {:.3f}'.format(f1_mean))
    print('Mean Relative Error: {:.3f}'.format(rel_err_mean))
    print('Mean Absolute Error: {:.3f}'.format(abs_err_mean))
    print('SAE: {:.3f}'.format(SAE_mean))


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    

torch.set_default_tensor_type(torch.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--dataset_code', type=str,
                    default='redd_lf', choices=['redd_lf', 'uk_dale'])
parser.add_argument('--validation_size', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--house_indicies', type=list, default=[1, 2, 3, 4, 5])
parser.add_argument('--appliance_names', type=list,
                    default=['microwave', 'dishwasher'])
parser.add_argument('--sampling', type=str, default='6s')
parser.add_argument('--cutoff', type=dict, default=None)
parser.add_argument('--threshold', type=dict, default=None)
parser.add_argument('--min_on', type=dict, default=None)
parser.add_argument('--min_off', type=dict, default=None)
parser.add_argument('--window_size', type=int, default=480)
parser.add_argument('--window_stride', type=int, default=400)
parser.add_argument('--normalize', type=str, default='mean',
                    choices=['mean', 'minmax'])
parser.add_argument('--denom', type=int, default=200)
parser.add_argument('--model_size', type=str, default='lstm',
                    choices=['gru', 'lstm', 'dae'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.25)
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str,
                    default='adam', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--c0', type=dict, default=None)
parser.add_argument('--model', type=dict, default=None)

args = parser.parse_args()


if __name__ == "__main__":
    fix_random_seed_as(args.seed)
    get_user_input(args)
    set_template(args)
    train(args)
