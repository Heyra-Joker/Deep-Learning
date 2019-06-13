import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.autograd import Variable

from LoadDataset import *
import matplotlib.pyplot as plt


class Forward(nn.Module):

    def __init__(self, rate):
        torch.manual_seed(123)
        nn.Module.__init__(self)
        self.CONV = nn.Sequential(
            self.CONV2D(3, 96, kernel_size=(11, 11), stride=(4, 4)),
            nn.ReLU(inplace=True),
            self.MAXPOOL(kernel_size=(3, 3), stride=(2, 2)),
            self.LRN(size=5),
            self.CONV2D(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            self.MAXPOOL(kernel_size=(3, 3), stride=(2, 2)),
            self.LRN(size=5),
            self.CONV2D(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            self.CONV2D(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            self.CONV2D(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            self.MAXPOOL(kernel_size=(3, 3), stride=(2, 2)),
        )

        self.FULLY = nn.Sequential(
            self.FC(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            self.DROPOUT(rate=rate),
            self.FC(4096, 4096),
            nn.ReLU(inplace=True),
            self.DROPOUT(rate=rate),

            self.FC(4096, 1000),
            nn.ReLU(inplace=True),
            self.DROPOUT(rate=rate),

            # output
            self.FC(1000, 1),
            nn.Sigmoid()
        )

    def CONV2D(self, in_channels, out_channels, kernel_size, stride, padding=0):

        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)

        nn.init.xavier_normal_(conv2d.weight)

        return conv2d

    def MAXPOOL(self, kernel_size, stride):

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)
        return maxpool

    def LRN(self, size, alpha=1e-4, beta=0.75, k=1):
        lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        return lrn

    def FC(self, in_features, out_features):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.xavier_normal_(fc.weight)
        return fc

    def DROPOUT(self, rate):
        dropout = nn.Dropout(p=rate)
        return dropout

    def forward(self, data):

        data = self.CONV(data)

        data = data.view(data.size(0), 256 * 6 * 6)

        data = self.FULLY(data)

        return data


def Score(model, cost, loader, epoch, epochs, type_data,device):
    model.eval()
    average_accuracy = 0
    average_loss = 0
    N = 0
    NN = 0
    with torch.no_grad():
        for img, label in loader:

            label = label.float().to(device)
            out = model.forward(img.to(device))
            loss = cost(out, label)
            predict = torch.round(out)
            equal_ = torch.eq(predict, label).float()
            accuracy = torch.mean(equal_).item()
            average_accuracy += accuracy
            average_loss += loss
            N += 1
            NN += img.size()[0]
            print('[{}/{}] scoring {} ==> {} loss:{} acc:{}\r'.format(epoch + 1, epochs, type_data, NN,loss
                                                                      ,accuracy), end='', flush=True)

        average_accuracy /= N
        average_loss /= N
        print()
        print('[{}/{}] {} average loss: {:.4f}, average accuracy: {:.4f}'.format(epoch + 1, epochs, type_data,
                                                                 average_loss, average_accuracy))

        return average_loss, average_accuracy


def Plot_loss_acc(loss, acc, lr, plt_path):
    figure = plt.figure(figsize=(10, 4))
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.plot(loss[0], '-o', label="train loss")
    ax1.plot(loss[1], '-o', c='orange', label='val loss')
    ax1.set_title('lr: {} train/test Loss'.format(lr))
    ax1.set_xlabel('#Iterate')
    ax1.set_ylabel('Values')
    ax1.legend()

    ax2 = figure.add_subplot(1, 2, 2)
    ax2.plot(acc[0], '-o', label="train accuracy")
    ax2.plot(acc[1], '-o', c='orange', label="val accuracy")
    ax2.set_title('lr: {} train/test Accuracy'.format(lr))
    ax2.set_xlabel('#Iterate')
    ax2.set_ylabel('Values')
    ax2.legend()

    plt.savefig(plt_path)


def AlexMoel(file_dir, lr, epochs, Load_samples=None, test_rate=0.3, drop_rate=0.3, batch_size=100, **kwargs):
    train_loss_list = []
    test_loss_list = []

    train_acc_list = []
    test_acc_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    split_data = SplitData(file_dir, Load_samples=Load_samples, test_rate=test_rate, Shuffle=True)
    train_files, test_files, train_samples, test_samples = split_data()

    print('Using {}, Train Samples:{} Test Samples:{}'.format(str(device), train_samples, test_samples))

    load_train = Load(train_files, transform=ToTensor())
    train_loader = DataLoader(load_train, batch_size=batch_size, shuffle=True, num_workers=2)

    load_test = Load(test_files, transform=ToTensor())
    test_loader = DataLoader(load_test, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Forward(drop_rate)
    if str(device) == 'cuda:0':
        model.cuda(device)

    cost = nn.BCELoss()
    optim = optimizer.RMSprop(model.parameters(), lr=lr, alpha=0.9)

    for epoch in range(epochs):
        model.train()
        NN = 0
        for img, label in train_loader:
            NN += img.size()[0]
            print('[{}/{}] train ==> [{}/{}] \r'.format(epoch+1,epochs,NN,train_samples),end='',flush=True)
            img = Variable(img).to(device)
            label = Variable(label).float().to(device)

            out = model.forward(img)
            loss = cost(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss, train_acc = Score(model, cost, train_loader, epoch, epochs, 'Train',device)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss, test_acc = Score(model, cost, test_loader, epoch, epochs, 'Test',device)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    plt_path = kwargs.get('plt_path', None)
    model_save_path = kwargs.get('model_save_path', None)

    if plt_path:
        print('Starting Plot Loss/Accuracy...')
        Plot_loss_acc((train_loss_list, test_loss_list), (train_acc_list, test_acc_list), lr, plt_path)
        print('PLot Finished --> {}'.format(plt_path))
    if model_save_path:
        print('Starting Save Model...')
        torch.save(model.state_dict(), model_save_path)
        print('Save Finished --> {}'.format(model_save_path))


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    AlexMoel(file_dir, lr=1e-4, epochs=13, Load_samples=100,
             test_rate=0.3, drop_rate=0.3, batch_size=10,
             plt_path='loss_acc.png',
             model_save_path='AlexNet_pytorch.pt')

