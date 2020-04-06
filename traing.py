import torch.nn as nn
import torch
from torch import nn
import numpy as np
import pandas as pd
import time
import json
from common import NumpyEncoder

torch.manual_seed(1)  # reproducible

# Hyper Parameters
TIME_STEP = 20  # rnn time step / image height
LR = 0.0002  # learning rate
BATCH_SIZE = 32


class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=input_size,
            hidden_size=50,  # rnn hidden unit
            num_layers=2,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            dropout=0.5
        )
        self.out = nn.Linear(50, 2)
        # self.out = nn.Linear(9, 2)

    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)

        # x(32,10,5)
        r_out, h_state = self.rnn(x, h_state)  # h_state 也要作为 RNN 的一个输入

        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


class train():
    output = 2
    def __init__(self):
        data = pd.read_csv("2w.csv").values[:-1, 1:].astype("float32")
        x_data = data[:, :-2]
        # 两两相乘
        # for i in range(data[:, :-2].shape[1]):
        #     for j in range(data[:, :-2].shape[1]):
        #         x1 = x_data[:, i].reshape(-1, 1)
        #         x2 = x_data[:, j].reshape(-1, 1)
        #         new_value = x1 * x2
        #         x_data = np.hstack([x_data, new_value])
        # #加入正弦和余弦
        #
        # x_data = np.hstack([x_data, np.sin(data[:, :-2]), np.cos(data[:, :-2])])

        self.std = np.std(x_data, axis=0)
        self.mean = np.mean(x_data, axis=0)
        self.input_size = len(self.mean)
        self.data = np.hstack([x_data, data[:, -2:]])

    def get_test_data(self):
        x_input = self.data[10000:, :-2]
        y_label = self.data[10000:, -2:]

        print(x_input.shape)
        print(y_label.shape)

        # exit()
        batch_index = []
        normalized_train_data = (x_input - self.mean) / self.std  # 标准化
        train_x, train_y = [], []  # 训练集x和y初定义
        for i in range(len(normalized_train_data) - TIME_STEP):
            if i % BATCH_SIZE == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + TIME_STEP]
            y = y_label[i:i + TIME_STEP]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - TIME_STEP))
        return batch_index, np.array(train_x).astype("float32"), np.array(train_y).astype("float32")
    def get_train_data(self):
        x_input = self.data[:10000, :-2]
        y_label = self.data[:10000, -2:]
        print(x_input.shape)
        print(y_label.shape)

        batch_index = []
        normalized_train_data = (x_input - self.mean) / self.std  # 标准化
        train_x, train_y = [], []  # 训练集x和y初定义
        for i in range(len(normalized_train_data) - TIME_STEP):
            if i % BATCH_SIZE == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + TIME_STEP]
            y = y_label[i:i + TIME_STEP]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - TIME_STEP))
        return batch_index, np.array(train_x).astype("float32"), np.array(train_y).astype("float32")

    def every_step(self, train_x, train_y, batch_index, IS_TEST=False):
        if IS_TEST:
            self.rnn.eval()
        else:
            self.rnn.train()
        h_state = None  # 要使用初始 hidden state, 可以设成 None
        total_loss = []
        for step in range(len(batch_index) - 2):
            x = torch.from_numpy(
                train_x[batch_index[step]: batch_index[step + 1]]).cuda()  # shape (batch, time_step, input_size)

            y = torch.from_numpy(train_y[batch_index[step]: batch_index[step + 1]]).cuda()
            prediction, h_state = self.rnn(x, h_state)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
            # !!  下一步十分重要 !!
            h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
            loss = self.loss_func(prediction, y)  # cross entropy loss
            total_loss.append(loss.data.cpu().numpy())
            if IS_TEST:
                continue
            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
        return np.mean(total_loss)

    def run(self):
        self.rnn = RNN(self.input_size)

        # self.rnn = torch.load('rnn.pkl')
        print(self.rnn)

        self.rnn.cuda()
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=LR)  # optimize all rnn parameters
        self.loss_func = nn.MSELoss()

        batch_index, train_x, train_y = self.get_train_data()
        batch_test, test_x, test_y = self.get_test_data()
        handle = open("train.log", "a")

        for i in range(9000000):
            total_loss = self.every_step(train_x, train_y, batch_index, False)
            test_loss = self.every_step(test_x, test_y, batch_test, True)
            message = "第:%s回合,loss:%s,test:%s" % (i, total_loss, test_loss)
            handle.writelines(message + "\n")
            print(message)
            torch.save(self.rnn, 'rnn.pkl')  # 保存整个网络


if __name__ == '__main__':
    train().run()
