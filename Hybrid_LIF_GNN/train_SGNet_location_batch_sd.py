# -*- coding: utf-8 -*`-
"""
Codes for Hybrid_LIF_GNN on event-driven slip detection v1 dataset, sd.

Codes are developed based on the work:
Gu, F., Sng, W., Taunyazov, T., & Soh, H. (2020). TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition,IROS 2020.

Please consider citing our work and the above work if you use the codes.

"""

from __future__ import print_function
import os
import time
from datetime import date
from torch.utils.data import Dataset
from tactileSGNet_location_batch_sd import *
import random
import tqdm
from pathlib import Path
import argparse


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

# model name
model_name = '_TactileSGNet_LOC_'  # tactile

# hyperparameter setting
parser = argparse.ArgumentParser("Train tactileSGNet_LOC models.")
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
num_classes = 2
k = 3  # number of nodes to be connected for constructing graph, not use
num_run = 1
learning_rate = 5.0*1e-3   # 1e-4 for cw&ycb
num_epochs = 150
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_length = 150


class tactileDataset(Dataset):
    def __init__(self, path, sample_file):
        self.path = path
        self.samples = np.loadtxt(Path(path) / sample_file).astype("int")
        tact = torch.load(Path(path) / "tact.pt")
        self.tact = tact

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]

        inputs = [self.tact[input_index]]

        return *inputs, torch.LongTensor([class_label])

    def __len__(self):
        return self.samples.shape[0]

# Decay learning rate
def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=30):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            if param_group['lr'] <= 1e-5:
                break
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# Tactile dataset
args = parser.parse_args()

trainDataset = tactileDataset(args.data_dir, f"train_80_20_{args.sample_file}.txt")
testDataset = tactileDataset(args.data_dir, f"test_80_20_{args.sample_file}.txt")
train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=False, num_workers=0)
# run for num_run times
best_acc = torch.zeros(num_run)
acc_list = list([])
training_loss_list = list([])
test_loss_list = list([])
net_list = list([])

for run in range(num_run):
    model = HybridLIFGNN(num_classes, time_length, k, device=device, sparse=True)
    model.to(device)
    criterion = nn.MSELoss()  # nn.MSELoss(reduction='sum') #nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.1)


    acc = torch.zeros(num_epochs)
    training_loss = torch.zeros(num_epochs)
    test_loss = torch.zeros(num_epochs)
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0
        for trainData, trainLabel in tqdm.tqdm(train_loader):

            model.zero_grad()
            optimizer.zero_grad()
            trainData = trainData.to(device)
            outputs = model(trainData)

            labels_ = torch.zeros(trainLabel.shape[0], num_classes).scatter_(1, trainLabel.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        training_loss[epoch] = running_loss

        # testing
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        running_loss = 0
        with torch.no_grad():
            for testData, testLabel in test_loader:

                optimizer.zero_grad()
                outputs = model(testData, False)
                labels_ = torch.zeros(testLabel.shape[0], num_classes).scatter_(1, testLabel.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                running_loss += loss.item()
                _, predicted = outputs.cpu().max(1)
                predicted = predicted.unsqueeze(1)
                # print(predicted.shape)
                total += float(testLabel.size(0))
                correct += float(predicted.eq(testLabel).sum().item())

            test_loss[epoch] = running_loss

            acc[epoch] = 100. * float(correct) / float(total)
            if best_acc[run] < acc[epoch]:
                best_acc[run] = acc[epoch]

        test_loss_list.append(test_loss)
        training_loss_list.append(training_loss)

        acc_list.append(acc)
        if (epoch + 1) % 2 == 0:
            print('At epoch: %s, training_loss: %f, test_loss: %f, best accuracy: %.3f, time elasped: %.3f' % (
            epoch + 1, training_loss[epoch], test_loss[epoch], best_acc[run], time.time() - start_time))
            start_time = time.time()
            if best_acc[run] == 100.0:
                break

    net_list.append(model.state_dict())

# overall state
state = {
    'net_list': net_list,
    'best_acc': best_acc,
    'num_epochs': num_epochs,
    'acc_list': acc_list,
    'training_loss_list': training_loss_list,
    'test_loss_list': test_loss_list,
}
dateStr = date.today().strftime("%Y%m%d")

if not os.path.isdir('log_data_TSG_LOC'):
    os.mkdir('log_data_TSG_LOC')
torch.save(state, './log_data_TSG_LOC/' + dateStr + model_name + '_objects_' + str(num_classes) + '_k_' + str(k) + '_file_' + str(args.sample_file) + '.t7')
print('Avg acc: %f, std: %f: ' % (torch.mean(state['best_acc']), torch.std(state['best_acc'])))