from dataset import data_dataloader  # 电脑本地写的读取数据的函数
from torch import nn  # 导入pytorch的nn模块
from torch import optim  # 导入pytorch的optim模块
from network import Res_net  # 网络框架
from train import train  # 训练函数


def main():
    # 以下是通过Data_dataloader函数输入为：数据的路径，数据模式，数据大小，batch的大小，有几线并用 （把dataset和Dataloader功能合在了一起）
    train_loader = data_dataloader(data_path='./data/train', mode='train', size=64, batch_size=24, num_workers=0)
    val_loader = data_dataloader(data_path='./data/val', mode='val', size=64, batch_size=24, num_workers=0)
    test_loader = data_dataloader(data_path='./data/test', mode='test', size=64, batch_size=24, num_workers=0)

    # 以下是超参数的定义
    lr = 1e-4  # 学习率
    epochs = 10  # 训练轮次

    model = Res_net(2)  # resnet网络
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
    loss_function = nn.CrossEntropyLoss()  # 损失函数

    # 训练以及验证测试函数
    train(model=model, optimizer=optimizer, loss_function=loss_function, train_data=train_loader, val_data=val_loader,
          test_data=test_loader, epochs=epochs)


if __name__ == '__main__':
    main()
