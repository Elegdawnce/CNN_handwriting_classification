import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import model
import utils


def train():
    # 加载模型
    cnn = model.CNN()
    print(cnn)

    # 获取计算设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            cnn = nn.DataParallel(cnn)
        print('Using %d GPU...' % num_gpu)
    else:
        device = torch.device('cpu')
        print('Using CPU...')
    # 网络转移到设备上
    cnn.to(device)


    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(cnn.parameters(), lr=dataset.LR, weight_decay=dataset.WEIGHT_DECAY)
    # 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, dataset.LR_DECAY_STEP_SIZE)

    for epoch in range(dataset.EPOCH):
        scheduler.step()

        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()

        # 训练一个epoch
        cnn.train()
        for data in dataset.train_loader:
            # 一个step，获取到一个batch
            x,y = data
            # 转移到设备上
            x,y = x.to(device),y.to(device)

            # 前向传播
            probs = cnn(x)[0]
            loss = criterion(probs,y)
            acc = utils.accuracy(y, probs)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()

            # 计算一个step 的 平均loss 和 平均acc
            train_loss.update(loss.item(),dataset.BATCH_SIZE)
            train_acc.update(acc)

        print('Train: Epoch:',epoch,'\tloss:%.4f' % train_loss.avg,'\tacc:%.2f%%' % train_acc.avg)

        torch.save(cnn.state_dict(),'./save_state_dict/state_dict.pkl')



    # 训练结束，进行测试
    if isinstance(cnn, nn.DataParallel):
        cnn.module.load_state_dict(torch.load('./save_state_dict/state_dict.pkl'))
    else:
        cnn.load_state_dict(torch.load('./save_state_dict/state_dict.pkl'))

    test_acc = utils.AverageMeter()
    count = 0

    cnn.eval()
    with torch.no_grad():
        for data in dataset.test_loader:
            x, y = data
            # 转移到设备上
            x,y=x.to(device),y.to(device)

            probs = cnn(x)[0]
            acc = utils.accuracy(y,probs)
            test_acc.update(acc.item(),dataset.BATCH_SIZE)

            y = y.type(torch.float32)
            predicted_y = (torch.max(probs, 1)[1]).type(torch.float32)
            while count < 10:
                print('True number:',y[count],'\tPredict number:',predicted_y[count])
                count += 1

    print('Test: accuracy:%.2f%%' % test_acc.avg)



if __name__ == '__main__':
    train()




