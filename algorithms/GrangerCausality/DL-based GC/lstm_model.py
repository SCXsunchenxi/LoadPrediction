import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,device):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device=device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # 第一层为LSTM

        self.fc = nn.Linear(hidden_size, num_classes)  # 第二层为一个全连接层
        print("lstm")

    def forward(self, x):
        # h_n是一个三维的张量,第一维是num_layers*num_directions。第二维表示一批的样本数量(batch)。第三维表示隐藏层的大小。
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0))  # out torch.Size([30736, 8, 256])  hn torch.Size([4, 30736, 256])
        # print("out的输出：", out.shape)
        # h_out = h_out.view(-1, self.hidden_size)  # view()方法可以调整tensor的形状,当某一维是-1时，会自动计算它的大小
        out = out[:, -1, :]  # torch.Size([11590, 21])
        out = self.fc(out)  # torch.Size([11590, 1501])
        return out