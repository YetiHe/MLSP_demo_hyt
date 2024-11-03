import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

best_train_acc = 0
best_val_acc = 0
best_test_acc = 0
best_acc = 0
# 这里是自定义的工具包，包含必要但是不重要的小函数。
# 改分割任务和预测任务后需要大改

def mkdir(path):  # 确认是否有地址，否则新建该地址文件
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def save_checkpoint(state, is_best, path, filename='last'):  # 存入当前进度，从而允许使用者随时停止，下次接着上次的进度继续
    name = os.path.join(path, filename + '_checkpoint.pth')
    print(name, '\n')
    torch.save(state, name)
    # torch.save()调包保存当前进度与模型参数

# 保存模型，存在checkpoint进度的文件那里，其实就是把上一个函数包装了一下，防止重复造车轮
# metrics是自定义的一个对象，其类在本文件的下面
def save_model(model,  args, metrics, best_pred_loss):
    global best_acc
    loss = metrics.data['loss']
    save_path = args.save
    mkdir(save_path)

    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    is_best = False
    if best_acc < metrics.data['dice'] / metrics.cnt:
        is_best = True
        best_pred_loss = loss
        best_acc = metrics.data['dice'] / metrics.cnt
        save_checkpoint({'state_dict': model.state_dict(),
                         'metrics': metrics.data},
                        is_best, save_path, "best")

    save_checkpoint({'state_dict': model.state_dict(),
                         'metrics': metrics.data},
                        is_best, save_path, "last")

    return best_pred_loss

def accuracy(output, target):
    with torch.no_grad():
        predict = (output >= 0.5)

        pre_1_num = predict.sum().item() # TP + FP
        label_1_num = target.sum().item() # TP + TN
        intersection = (predict * target).sum().item() # TP
        tmp = predict + target
        tmp[tmp > 1] = 1
        union = tmp.sum().item() # TP + FP + TN

        accu = (predict == target).sum().item() / target.view(-1).size(0)
        dice = 2*intersection / (label_1_num+pre_1_num)
        precision = intersection / pre_1_num
        IOU = intersection/union
        recall = intersection / label_1_num

        TP = intersection
        TN = label_1_num - TP
        FP = union - TP - TN
        FN = target.view(-1).size(0) - TP - TN - FP

        confusion_param = np.asarray([TP, TN, FP, FN]) / target.view(-1).size(0)

    return accu, precision, dice, IOU, recall, confusion_param

# 存储训练信息或测试信息的类，集成了各种数据分析用的小函数。
class Metrics:
    def __init__(self, path, keys=None, writer=None):
        self.writer = writer
        self.cnt = 0
        self.data = {'dice': 0,
                     'IOU': 0,
                     'loss': 0,
                     'accuracy': 0,
                     'precision': 0,
                     'recall': 0
                     }
        self.confusion_param = np.zeros((4,1))


    def reset(self):
        for key in self.data:
            self.data[key] = 0
        self.cnt = 0

    def update(self, values, cf_param=None):
        for key in self.data:
            self.data[key] += values[key]
        self.cnt += 1
        if cf_param is not None:
            self.confusion_param += cf_param.reshape((4,1))

    def draw_confusion(self):
        cf_param = self.confusion_param / self.cnt
        cf_matrix = np.zeros((2,2))
        cf_matrix[0,0] = cf_param[0]
        cf_matrix[0,1] = cf_param[1]
        cf_matrix[1,0] = cf_param[2]
        cf_matrix[1,1] = cf_param[3]

        plt.figure(1)
        plt.imshow(cf_matrix,  cmap='viridis')
        plt.title("Confusion Matrix")
        plt.colorbar()

        plt.xticks([0, 1],['P', 'N'])
        plt.yticks([0, 1],['T', 'F'])

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 在图中显示数值
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cf_matrix[i, j], horizontalalignment="center",
                         color="white" if cf_matrix[i, j] > cf_matrix.max() / 2 else "black")

        plt.savefig('confusion_tmp.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()


# 打印所有你觉得要看看的信息，同时更新最好的数据
def print_summary(args, epoch, metrics, mode=''):
    global best_train_acc
    global best_test_acc
    global best_val_acc
    if mode == 'train':
        if best_train_acc < 100. * metrics.data['dice'] / metrics.cnt:
            best_train_acc = 100. * metrics.data['dice'] / metrics.cnt
        print(
            mode + "\tEPOCH:{:2d}/{:3d}\t\tLoss:{:.6f}\tAcc:{:.2f}\tDice:{:.2f}\tIOU:{:.2f}\tPrecision:{:.2f}\tRecall:{:.2f}\tBest dice:{:.2f}\n".format(
                epoch, args.nEpochs,
                metrics.data['loss'] / metrics.cnt,
                metrics.data['accuracy']/metrics.cnt,
                metrics.data['dice'] / metrics.cnt,
                metrics.data['IOU'] / metrics.cnt,
                metrics.data['precision'] / metrics.cnt,
                metrics.data['recall'] / metrics.cnt,
                best_train_acc
            ))
    if mode == 'test':
        if best_test_acc < 100. * metrics.data['dice'] / metrics.cnt:
            best_test_acc = 100. * metrics.data['dice'] / metrics.cnt
        print(
            mode + "\tEPOCH:{:2d}/{:3d}\t\tLoss:{:.6f}\tAcc:{:.2f}\tDice:{:.2f}\tIOU:{:.2f}\tPrecision:{:.2f}\tRecall:{:.2f}\tBest dice:{:.2f}\n".format(
                epoch, args.nEpochs,
                metrics.data['loss'] / metrics.cnt,
                metrics.data['accuracy'] / metrics.cnt,
                metrics.data['dice'] / metrics.cnt,
                metrics.data['IOU'] / metrics.cnt,
                metrics.data['precision'] / metrics.cnt,
                metrics.data['recall'] / metrics.cnt,
                best_test_acc
            ))
