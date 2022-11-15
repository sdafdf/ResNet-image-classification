import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import Dataset_self
from network import Res_net
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


def evaluate(model, loader):  # 计算每次训练后的准确率
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)  # 得到logits中分类值（要么是[1,0]要么是[0,1]表示分成两个类别）
        correct += torch.eq(pred, y).sum().float().item()  # 用logits和标签label想比较得到分类正确的个数
    return correct / total


# 把训练的过程定义为一个函数
def train(model, optimizer, loss_function, train_data, val_data, test_data, epochs):  # 输入：网络架构，优化器，损失函数，训练集，验证集，测试集，轮次
    best_acc, best_epoch = 0, 0  # 输出验证集中准确率最高的轮次和准确率
    train_list, val_List = [], []  # 创建列表保存每一次的acc，用来最后的画图
    for epoch in range(epochs):
        print('============第{}轮============'.format(epoch + 1))
        for steps, (x, y) in enumerate(train_data):  # for x,y in train_data
            logits = model(x)  # 数据放入网络中
            loss = loss_function(logits, y)  # 得到损失值
            optimizer.zero_grad()  # 优化器先清零，不然会叠加上次的数值
            loss.backward()  # 后向传播
            optimizer.step()
        train_acc = evaluate(model, train_data)
        train_list.append(train_acc)
        print('train_acc', train_acc)
        # if epoch % 1 == 2:   #这里可以设置每两次训练验证一次
        val_acc = evaluate(model, val_data)
        print('val_acc=', val_acc)
        val_List.append((val_acc))
        if val_acc > best_acc:  # 判断每次在验证集上的准确率是否为最大
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.mdl')  # 保存验证集上最大的准确率
    print('===========================分割线===========================')
    print('best acc:', best_acc, 'best_epoch:', best_epoch)
    # 在测试集上检测训练好后模型的准确率
    model.load_state_dict((torch.load('best.mdl')))
    print('detect the test data!')
    test_acc = evaluate(model, test_data)
    print('test_acc:', test_acc)
    train_list_file = np.array(train_list)
    np.save('train_list.npy', train_list_file)
    val_list_file = np.array(val_List)
    np.save('val_list.npy', val_list_file)

    # 画图
    x_label = range(1, len(val_List) + 1)
    plt.plot(x_label, train_list, 'bo', label='train acc')
    plt.plot(x_label, val_List, 'b', label='validation acc')
    plt.title('train and validation accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


# 测试
def main():
    train_dataset = Dataset_self('./data', 'train', 64)
    vali_dataset = Dataset_self('./data', 'val', 64)
    test_dataset = Dataset_self('./data', 'test', 64)

    train_loaber = DataLoader(train_dataset, 24, num_workers=0)
    val_loaber = DataLoader(vali_dataset, 24, num_workers=0)
    test_loaber = DataLoader(test_dataset, 24, num_workers=0)

    lr = 1e-4
    epochs = 5
    model = Res_net(2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    train(model, optimizer, criteon, train_loaber, val_loaber, test_loaber, epochs)


if __name__ == '__main__':
    main()
