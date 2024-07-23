import torch
import torch.nn as nn

from args import Args

args = Args().parse()

"""
# 2048-->1024-->512-->40/50
class Teacher(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(Teacher, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            self.bn1,
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            self.bn2,
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x):
        output = self.net(x)
        return output
"""


class Teacher(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(Teacher, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.lRelu = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dt = nn.Dropout(p=dropout)
        # self.hidden2 = hidden_dim2
        # self.hidden = self.init_hidden()

    """ 
    def obtain_center(self, x):
        inter_output = self.fc2(self.dt(self.lRelu(self.bn1(self.fc1(x)))))  # (batch,512)
        return inter_output
    """
    """
    def init_hidden(self):
        return (torch.zeros(args.batchTS, self.hidden2))
    """

    def forward(self, x):
        # inter_output = self.init_hidden()
        data = self.fc1(x)
        data = self.bn1(data)
        data = self.lRelu(data)
        data = self.dt(data)
        data = self.fc2(data)
        data = self.bn2(data)
        data = self.lRelu(data)
        data = self.dt(data)
        output = self.fc3(data)
        #inter_output = self.fc2(self.dt(self.lRelu(self.bn1(self.fc1(x)))))  # (batch,512)
        #output = self.fc3(self.dt(self.lRelu(self.bn2(inter_output))))  # (batch, 50/40)
        return output


# center loss method 1
# 1.Train teacher: hidden_layer2(batch_size,512)-->center loss + ctn_loss
# 2.Get Teacher's hidden_layer2(batch_size, 512)-->get mean 512 according to each cls
# 3.Train generator-->2048-->teacher--> crossEntropyloss + [each cls: (batch_size,512)-mean 512]

# 1.训练完teacher，把数据按一个batch送进去，得到（16000，512）
# 2.（16000，512）按类别计算中心点（按类别求个平均的512），并存取（50/40，512）
# 3. 训练generator：768+nl--》1024--》2048--》teacher（取fc2的512（batch，512）
# 遍历batch，按类别将每个512-第二部中的512）【20，512】--》mse（【512】-【512】）+【20】loss：item

# 768+nl -->1024--2048
class Generator(nn.Module):
    def __init__(self, att_dim, noise_dim, hidden_dim, output_dim, dropout=0.2):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.input_dim = att_dim + noise_dim if args.noise_type != 'p_noises' else args.noiseLen
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.restrictions = args.noise_type
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.bn1,
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            self.bn2,
            nn.ReLU(),
        )

    def add_noise(self, x):
        if args.noise_type != 'p_noises':
            _x = torch.zeros([1, x.size(1) + self.noise_dim], dtype=torch.float32)
            if self.restrictions == 'extremum_dis':
                for i in range(x.size(0)):
                    min_v = x[i].min()
                    max_v = x[i].max()
                    noise = torch.Tensor(self.noise_dim, ).uniform_(min_v, max_v).to(args.cuda_device)
                    _x_ = torch.hstack((x[i], noise)).view(-1, x.size(1) + self.noise_dim)
                    _x = _x_ if i == 0 else torch.cat((_x, _x_), 0)
            else:
                for i in range(x.size(0)):
                    noise = torch.rand(self.noise_dim, ).to(
                        args.cuda_device) if self.restrictions == 'evenly_dis' else torch.randn(
                        self.noise_dim, ).to(args.cuda_device)
                    _x_ = torch.hstack((x[i], noise)).view(-1, x.size(1) + self.noise_dim)
                    _x = _x_ if i == 0 else torch.cat((_x, _x_), 0)

            _x.requires_grad = True
            return _x
        else:
            _x = torch.zeros([1, self.input_dim], dtype=torch.float32)
            for i in range(args.batchTS):
                pure_noise = torch.rand(1, self.input_dim).to(
                    args.cuda_device) if self.restrictions == 'evenly_dis' else torch.randn(1, self.input_dim).to(
                    args.cuda_device)
                _x = pure_noise if i == 0 else torch.cat((_x, pure_noise), 0)
            _x.requires_grad = True
            return _x

    def forward(self, x):
        output = self.add_noise(x)
        output = self.net(output)
        return output


class Student(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.2):
        super(Student, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            # add one liner layer
            nn.Linear(hidden_dim2, args.hidSizeTSZ_3),
            nn.BatchNorm1d(args.hidSizeTSZ_3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            #
            #nn.Linear(hidden_dim2, output_dim),
            nn.Linear(args.hidSizeTSZ_3, output_dim),
        )

    def forward(self, x):
        output = self.net(x)
        return output


class Z_net(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(Z_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        output = self.net(x)
        return output
