import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from loadData_Torch import SplitData, LoadData, ToTensor, DataLoader


def init_weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


class ZFNet(nn.Module):
    def __init__(self, rate, num_classes=1):

        torch.manual_seed(seed=8)
        nn.Module.__init__(self)
        self.Conv = nn.ModuleList([
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            # Conv2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            # Conv3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            # Conv4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            # Conv5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)
        ])
        self.FC = nn.ModuleList([
            # FC6
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=rate),
            # FC7
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=rate),
            # out
            nn.Linear(in_features=4096, out_features=num_classes)
        ])

    def forward(self, x):
        for conv in self.Conv:
            x = conv(x)
        x = x.view(-1, 6 * 6 * 256)
        for fc in self.FC:
            x = fc(x)
        return x


class ZFModel:
    def __init__(self, file_dir, Load_samples=100, test_rate=0.3, BATCH_SIZE=50):

        self.file_dir = file_dir
        self.Load_samples = Load_samples
        self.test_rate = test_rate
        self.BATCH_SIZE = BATCH_SIZE

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def split(self):
        split_data = SplitData(self.file_dir, Load_samples=self.Load_samples, test_rate=self.test_rate)
        train_files, test_files = split_data()

        self.N_train = len(train_files) // self.BATCH_SIZE
        self.N_test = len(test_files) // self.BATCH_SIZE
        return train_files, test_files

    def loader(self, files):
        loader_ = LoadData(files, ToTensor())
        _loader = DataLoader(loader_, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)
        return _loader

    def train(self, model, epoch, epochs, loader_train, optimizer, Cost):
        model.train()
        _loss = 0
        _correct = 0
        for index, (image, labels) in enumerate(loader_train):
            print('[{}/{}] Train on [{}/{}] \r'.format(epoch + 1, epochs, index + 1, self.N_train), end="",
                  flush=True)
            image, labels = image.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            out = model(image)
            loss = Cost(out, labels)
            # add loss
            _loss += loss.item()
            loss.backward()
            optimizer.step()

            predict = torch.argmax(F.log_softmax(out, dim=1), dim=1)
            # calculate correct.
            _correct += predict.eq(labels.view_as(predict)).sum().item()

        _loss /= len(loader_train.dataset)
        _correct /= len(loader_train.dataset)

        print('=>Train EPOCH:[{}/{}]  loss:{:.4f} acc:{:.4f}'.format(epoch + 1, epochs, _loss, _correct))

    def test(self, model, loader_test, epoch, epochs, Cost):
        model.eval()
        _loss = 0
        _correct = 0

        with torch.no_grad():
            for index, (image, labels) in enumerate(loader_test):
                print('testing on [{}/{}] \r'.format(index + 1, self.N_test), end="", flush=True)
                image, labels = image.to(self.device), labels.to(self.device)

                out = model(image)
                # sum up batch size.
                _loss += Cost(out, labels).item()
                predict = torch.argmax(F.log_softmax(out, dim=1), dim=1)
                # calculate correct.
                _correct += predict.eq(labels.view_as(predict)).sum().item()

            _loss /= len(loader_test.dataset)
            _correct /= len(loader_test.dataset)

            print('~>Test EPOCH:[{}/{}] loss:{:.4f} accuracy:{:.4f}'.format(epoch + 1, epochs, _loss, _correct))

            return _correct

    def fit(self, rate, num_classes, lr, epochs, save_model=None):
        print('|<==>| Pytorch Using {} |<==>|'.format(self.device))

        train_files, test_files = self.split()

        _loader_train = self.loader(train_files)
        _loader_test = self.loader(test_files)

        model = ZFNet(rate=rate, num_classes=num_classes)
        model.apply(init_weight)
        model.to(self.device)
        Cost = nn.CrossEntropyLoss(reduction='sum')
        optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)

        for epoch in range(epochs):
            self.train(model, epoch, epochs, _loader_train, optimizer, Cost)
            _correct = self.test(model, _loader_test, epoch, epochs, Cost)
            if _correct >= 0.98:
                break
        # saver...
        if save_model:
            torch.save(model.state_dict(), save_model)


if __name__ == "__main__":
    file_dir = '/Users/huwang/Joker/Data_Set/catVSdot/train'
    save_model = 'ZFNet.pt'
    zf = ZFModel(file_dir, Load_samples=20, test_rate=0.2, BATCH_SIZE=10)
    zf.fit(rate=0.5, num_classes=2, lr=1e-4, epochs=2, save_model=None)
