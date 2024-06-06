import argparse
import os
import time
import pickle
import torch
from sklearn import model_selection
from torch import nn
from torch.utils.data import DataLoader
import sys

sys.path.append(os.getcwd())
from final_model import FinalModel
from earlystopping import EarlyStopping
from Logger import Logger
from mydataset import MyDataset

sys.stdout = Logger(filename="训练情况.txt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_load(data_dir):
    data_hou = "all_sub_data.pkl"
    label_hou = "all_label.pkl"
    data = pickle.load(open(data_dir + data_hou, "rb"))
    labels = pickle.load(open(data_dir + label_hou, "rb"))
    eeg_data = data[:, :, 0:32]
    eda_data = data[:, :, 36:37]
    ppg_data = data[:, :, 38:39]
    eeg_data = eeg_data.astype('float32')
    eda_data = eda_data.astype('float32')
    ppg_data = ppg_data.astype('float32')
    labels = labels.astype('int64')
    return eda_data, ppg_data, eeg_data, labels


def train(dataloader, model, loss_fn, optimizer, scheduler):
    train_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    for batch, (eda_data, ppg_data, eeg_data, label) in enumerate(dataloader, 0):
        eda_data = eda_data.to(device)
        ppg_data = ppg_data.to(device)
        eeg_data = eeg_data.to(device)
        label = label.to(device)
        pred = model(eda_data, ppg_data, eeg_data)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(eda_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()
    return train_loss / len(dataloader)


def valid(dataloader, model, loss_fn, early_stopping):
    print("开始验证\n-------------------------------")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for eda_data, ppg_data, eeg_data, label in dataloader:
            eda_data, ppg_data, eeg_data, label = eda_data.to(device), ppg_data.to(device), eeg_data.to(
                device), label.to(device)
            pred = model(eda_data, ppg_data, eeg_data)
            loss = loss_fn(pred, label).item()
            test_loss += loss
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"valid : \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print("早停了！")
        return True
    return False


def test(dataloader, model, loss_fn):
    print("开始测试\n-------------------------------")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for eda_data, ppg_data, eeg_data, label in dataloader:
            eda_data, ppg_data, eeg_data, label = eda_data.to(device), ppg_data.to(device), eeg_data.to(
                device), label.to(device)
            pred = model(eda_data, ppg_data, eeg_data)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test : \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    with open("test.txt", "w") as f:
        f.write(f"Test : \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    current_file_abs_path = os.getcwd()

    '''
        设置超参数
    '''
    parser = argparse.ArgumentParser(description="Net")

    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")

    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--gpus', default=0, type=int, nargs='+')

    parser.add_argument('--lam_regularize', default=0.0, type=float,)

    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--modelpath', default='checkpoint.pt', type=str)
    args = parser.parse_args()

    print("epochs:", args.epochs)
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("gpus:", args.gpus)
    print("lam_regularize:", args.lam_regularize)

    print(time.asctime(time.localtime(time.time())))

    data_dir = current_file_abs_path + "\\data\\"
    eda_data, ppg_data, eeg_data, labels = data_load(data_dir)
    print("eda_data.shape:", eda_data.shape)
    print("ppg_data.shape:", ppg_data.shape)
    print("eeg_data.shape:", eeg_data.shape)
    print("labels.shape:", labels.shape)

    eda_train, eda_test, ppg_train, ppg_test, eeg_train, eeg_test, y_train, y_test = model_selection.train_test_split(
        eda_data, ppg_data,
        eeg_data, labels,
        train_size=0.9, stratify=labels, random_state=42)
    eda_train, eda_valid, ppg_train, ppg_valid, eeg_train, eeg_valid, y_train, y_valid = model_selection.train_test_split(
        eda_train, ppg_train,
        eeg_train, y_train,
        train_size=0.9, stratify=y_train, random_state=42)

    print("eda_train.shape:", eda_train.shape)
    print("ppg_train.shape:", ppg_train.shape)
    print("eeg_train.shape:", eeg_train.shape)
    print("y_train.shape:", y_train.shape)
    print("eda_valid.shape:", eda_valid.shape)
    print("ppg_valid.shape:", ppg_valid.shape)
    print("eeg_valid.shape:", eeg_valid.shape)
    print("y_valid.shape:", y_valid.shape)
    print("eda_test.shape:", eda_test.shape)
    print("ppg_test.shape:", ppg_test.shape)
    print("eeg_test.shape:", eeg_test.shape)
    print("y_test.shape:", y_test.shape)
    with torch.no_grad():

        train_dataset = MyDataset(eda_train, ppg_train, eeg_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = MyDataset(eda_valid, ppg_valid, eeg_valid, y_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = MyDataset(eda_test, ppg_test, eeg_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FinalModel(device)

    E_loss = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lam_regularize)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 * (1.0 ** epoch))

    estop = EarlyStopping(patience=args.patience, verbose=True, path=args.modelpath)

    print("-" * 30, "训练模型", "-" * 30)

    epochs = args.epochs
    for t in range(epochs):
        train_start_time = time.time()
        print(f"Epoch {t}\n-------------------------------")
        train_loss = train(train_dataloader, model, E_loss, optimizer, scheduler)
        train_used_time = time.time() - train_start_time
        print(f'epoch {t} time: {train_used_time} epoch_loss {train_loss}\n-------------------------------')
        x_len = t + 1
        if not os.path.exists(current_file_abs_path + '\\checkpoints\\'):
            os.makedirs(current_file_abs_path + '\\checkpoints\\')
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, current_file_abs_path + f'\\checkpoints\\model_epoch{t}.pt')

        is_stop = valid(valid_dataloader, model, E_loss, estop)
        if is_stop:
            break
        torch.cuda.empty_cache()

    model.load_state_dict(torch.load(f'./{args.modelpath}'))
    print("-" * 30, "测试模型", "-" * 30)
    test(test_dataloader, model, E_loss)
    print("Done!")
