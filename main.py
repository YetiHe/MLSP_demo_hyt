import warnings
warnings.filterwarnings('ignore')  # 这个warning没有用，所以ignore掉
import argparse
import numpy as np
import util
from train import *
import time
import math


def adjust_learning_rate(optimizer, epoch, args):  # 调整学习速率
    lr_min = 0
    if args.cosine:
        # lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi / args.nEpochs)) * (args.lr - lr_min) / 2.)
        lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi / 36)) * (args.lr - lr_min) / 2.)
    else:
        lr = args.lr * (0.1 ** (epoch // 30)) # 普通的衰减 每个epoch用一次
    print("learning rate is %f "% lr )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 更新亚当优化器的学习速率

def main():
    # 读入参数
    args = get_arguments()
    SEED = args.seed # 拿到随机种子
    np.random.seed(SEED) # 固定np的随机种子
    torch.manual_seed(SEED) # 固定torch的随机种子
    # torch.backends.cudnn.deterministic = True  # 优化的函数，可以锁定参数，效果更好
    # torch.backends.cudnn.benchmark = False  #

    if torch.cuda.is_available():
        print("using cuda")
        torch.cuda.manual_seed(SEED)

    # 确认日记地址是否存在，不存在就新建一个
    util.mkdir('runs')

    # 输入日记地址，待会保存记录用
    test_acc_file = 'runs/test.txt'
    train_acc_file = 'runs/train.txt'
    open(test_acc_file, 'w')
    open(train_acc_file, 'w')

    print('Building model, loading data...\n')
    # 初始化提前保存好的输入参数，得到需要的模型、优化器、训练集loader与测试集loader
    model, optimizer, training_generator, test_generator = initialize(args)  # 初始化所有参数，改训练策略只需要在此函数内改，从而操纵性提高

    best_pred_loss = 1000.0  # 记录最优的预测损失值
    print('\nCheckpoint folder:', args.save)

    # 主体：训练
    criterion = torch.nn.BCELoss(reduction="mean")
    for epoch in range(1, args.nEpochs + 1):  # 按照epoch进行训练，一个epoch是遍历了一次完整数据集，一般epoch至少要两位数
        train_metrics = train(args, model, training_generator, optimizer, epoch, criterion)
        test_metrics, ucsd_correct_total, ucsd_test_total = validation(args, model, test_generator, epoch, criterion)
        best_pred_loss = util.save_model(model, args, test_metrics, best_pred_loss)

        with open(test_acc_file, 'a+') as f:  # 写入测试日记保存记录
            f.write(str(test_metrics.data['accuracy'] / test_metrics.cnt) + ' ' +
                    str(test_metrics.data['dice'] / test_metrics.cnt) + ' ' +
                    str(test_metrics.data['IOU'] / test_metrics.cnt) + ' ' +
                    str(test_metrics.data['precision'] / test_metrics.cnt) + ' ' +
                    str(test_metrics.data['recall'] / test_metrics.cnt) + ' ' +
                    str(test_metrics.data['loss'] / test_metrics.cnt) + ' ' +
                    str(optimizer.param_groups[0]['lr']) + '\n')
        with open(train_acc_file, 'a+') as f:  # 写入训练日记保存记录
            f.write(str(train_metrics.data['accuracy'] / train_metrics.cnt) + ' ' +
                    str(train_metrics.data['dice'] / train_metrics.cnt) + ' ' +
                    str(train_metrics.data['IOU'] / train_metrics.cnt) + ' ' +
                    str(train_metrics.data['precision'] / train_metrics.cnt) + ' ' +
                    str(train_metrics.data['recall'] / train_metrics.cnt) + ' ' +
                    str(train_metrics.data['loss'] / train_metrics.cnt) + ' ' +
                    str(optimizer.param_groups[0]['lr']) + '\n')

        adjust_learning_rate(optimizer, epoch, args)  # 调整学习速率

def get_arguments():
    parser = argparse.ArgumentParser()
    # 以下是基本的训练策略
    parser.add_argument('--batch_size', type=int, default=16)  # batch_size为一批同时训练的数据的数量
    parser.add_argument('--nEpochs', type=int, default=12)  # epoch,一个epoch意味着遍历一次数据集

    # 以下是随机种子，可以固定一个种子以保证复现的时候准确率一致
    parser.add_argument('--seed', type=int, default=42)  # 用来打乱数据集排列用的随机数生成的随机种子，默认为643

    # 以下是网络输出的时候的维度，是2分类就输出2个输出，现在要改分割和预测
    parser.add_argument('--classes', type=int, default=2)  # 指的一共两个类

    # 学习速率
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-4)')  # 初始学习速率，默认为1E-4

    # 以下是亚当优化器需要的参数
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-7)')  # 权重衰减是调包torch.optim.adam要的，可以控制权重的值不要变动太大，拟合效果才会更好

    # 以下是数据集地址
    parser.add_argument('--dataset', type=str, default='..\\data',
                        help='path to dataset ')  # 数据集地址

    parser.add_argument('--cosine', default=True,
                        help='learning rate adjust scheme ')  # 学习速率的调整方案cosine是否开启

    # 以下是保存模型地址
    parser.add_argument('--save', type=str, default='saved\\cont',
                        help='path to checkpoint ')  # 训练到一半的保存地址，这样可以允许使用者随时停止，下次训练直接接上上次的进度

    parser.add_argument('--dim', type=tuple, default=[80, 80, 52])  # 要统一的输入尺寸

    parser.add_argument('--is_continue',type=bool,default=False,  # 接着上次的模型继续训练
                        help='continue training the last model')

    # 分类和预测任务用不上AUC策略
    parser.add_argument('--needROC',type=bool,default=False,
                        help='draw ROC and calculate AUC for each validation in an epoch')

    args = parser.parse_args()  # 返回初始化的各个参数
    return args


if __name__ == '__main__':
    since1 = time.time()
    main()
    done1 = time.time()
    print('Total Time:', done1 - since1, 's\n')
