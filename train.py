# coding=UTF-8
from util import Metrics, print_summary, accuracy
import dataset_MRI as ds
from tqdm import tqdm
from torch.utils.data import DataLoader
import model_MRI as md
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def initialize(args):
    model = md.simpleUNet(args.classes)  # 初始化模型（继承nn包后自定义）
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 用亚当优化器(调包)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.is_continue==True:
        if torch.cuda.is_available():
            model.load_state_dict(  # 载入此前已经训练好的模型
            torch.load('saved/cont/last_checkpoint.pth', map_location=str('cuda'))['state_dict'])
        else:
            model.load_state_dict(  # 载入此前已经训练好的模型
                torch.load('saved/cont/last_checkpoint.pth', map_location=str('cpu'))['state_dict'])

    # 设定训练和测试的参数，具体值在main.initialize()里已经定义过，这样统一设定再传入会更具有操纵性、更不易出错
    # 训练集随机打乱数据集，测试集不随机打乱数据集，两者用是同一个loader函数，
    # num_workers是处理数据的并行进程数量，为0则为用主进程load
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0}
    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 0}


    master = ds.Dataset_master({'dirpath': args.dataset, 'norm_size': args.dim}, ratio=0.8)
    train_set = master.train
    test_set = master.test

    training_loader = DataLoader(train_set, **train_params)
    test_loader = DataLoader(test_set, **test_params)

    return model, optimizer, training_loader, test_loader

# 注意此处的epoch在外面调用，不然很难打断进程。此train只负责训练一个epoch的内容，即把所有batch传一遍
def train(args, model, trainloader, optimizer, epoch,criterion):
    model.train()  # 此模型继承nn，pytorch里的父类自带train()函数，不需要重写，train的意思是改模式为train模式
    metrics = Metrics('')  # 初始化保存所有分析用信息的自定义类，因为此时还没信息产生，所以输入‘’
    metrics.reset()  # 初始化，防止残余内存干扰。
    batch_idx = 0  # 第几批

    # tqdm专门搞进度条显示用的，但是因为要显示进度条就要能指导迭代到哪了，所以tqpm其实有可以把可迭代对象的每个迭代元素抽出来的能力
    # 也因此，在这里用tqpm其实是为了抽取trainloader这个可迭代对象的内部元素，比如各种张量tensor（即数据本身）
    # trainloader 在这里是由调包实现的
    for input_tensors in tqdm(trainloader):
        batch_idx = batch_idx + 1
        optimizer.zero_grad()  # 调包亚当优化器的功能，用于梯度置零，
        input_data, labels = input_tensors  # 从抽取的元素中，读取所谓的输入数据、目标。因为是调包，所以估计这些会自动给
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            labels = labels.cuda()
        # print(torch.mean(labels))

        # 此处在训练！！！
        output = model(input_data)

        loss = criterion(output, labels)  # 这是外部定义的交叉熵梯度
        loss.backward()  # loss，这里是后向传播，其他的torch会自动解决
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # 防止梯度爆炸用的，2是个乘法因子
        optimizer.step()  # 亚当优化器推进！
        acc, precision, dice, iou, recall, confusion_param = accuracy(output, labels)  # 获取你想要的信息
        print(f"    batch {batch_idx} info: loss = {loss.item()}; acc = {acc}; dice = {dice}; recall = {recall}; precision = {precision}; IOU = {iou}")

        metrics.update({'dice': dice, 'IOU': iou, 'loss': loss.item(), 'accuracy': acc, 'precision':precision, 'recall':recall} ,confusion_param)
        # metrics.draw_confusion()

    print_summary(args, epoch, metrics, mode="train")
    metrics.draw_confusion()
    return metrics

# 需要大改
#  这是用于预测训练时的内部测试集标签的，从而确定其精确度。需要明确的时每个epoch都会传入一次训练集和一次测试集，从而获取每次epoch的性能，测试集并非是在epoch外才统一用
def validation(args, model, testloader, epoch, criterion, mode='test'):
    model.eval()  # 改测试状态
    # 基本上是重复之前的train操作
    metrics = Metrics('')
    metrics.reset()
    batch_idx = 0
    ucsd_correct_total = 0
    ucsd_test_total = 0

    with torch.no_grad():  # 意思是此with内的所有前向传播不会自动为后期潜在的后向传播计算导数，从而大大减少内存占用。验证时不需要后向传播，所以用此with torch.no_grad
        for input_tensors in tqdm(testloader):
            batch_idx = batch_idx + 1
            input_data, labels = input_tensors

            if torch.cuda.is_available():
                input_data = input_data.cuda()
                labels = labels.cuda()

            # if len(input_data) > 1:
            output = model(input_data)

            loss = criterion(output, labels)
            acc, precision, dice, iou, recall, confusion_param = accuracy(output, labels)  # 获取你想要的信息
            print(
                f"    batch {batch_idx} info: loss = {loss.item()}; acc = {acc}; dice = {dice}; recall = {recall}; precision = {precision}; IOU = {iou}")
            metrics.update({'dice': dice, 'IOU': iou, 'loss': loss.item(), 'accuracy': acc, 'precision': precision, 'recall':recall}, confusion_param)

    print_summary(args, epoch, metrics, mode="test")
    metrics.draw_confusion()

    # pred_label = record_label[2:record_label.shape[0]]
    # pred_postive = record_out[1:record_out.shape[0],1]
    # if args.needROC and args.nEpochs == epoch :
    #     fpr, tpr, thre = roc_curve(pred_label, pred_postive, pos_label=1)
    #     roc_auc = auc(fpr, tpr)
    #     plt.figure(figsize=(10, 10))
    #     plt.plot(fpr, tpr, color='black',
    #              label='ROC curve(area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1],
    #              color='red',
    #              linestyle='--')
    #     plt.xlabel("FPR, 1 - specificity")
    #     plt.ylabel("TPR, sensitivity")
    #     plt.grid()
    #     plt.title("ROC curve; AUC = %0.4f" % (roc_auc))
    #     plt.show()
#
    return metrics, ucsd_correct_total, ucsd_test_total
