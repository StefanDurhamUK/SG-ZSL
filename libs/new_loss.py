import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import Args
from pro_data import ProData

sys.path.append("libs")
args = Args().parse()


class NewLoss:
    def __init__(self):
        self.args = args
        self.device = self.args.cuda_device
        self.pro_data = ProData(self.args)
        # self.KL_loss = nn.KLDivLoss(size_average=True, reduce=True).to(self.device)
        self.teacher_train_dataloader, self.teacher_test_dataloader = self.obtain_real_data()

    def kl_loss(self, fake_features_batch):
        real_features_batch, real_labels_batch = next(iter(self.teacher_train_dataloader))
        log_ff = F.log_softmax(fake_features_batch, dim=-1)
        softmax_rf = F.softmax(real_features_batch, dim=-1)
        loss = F.kl_div(log_ff, softmax_rf, reduction='mean')
        return loss

    def mmd_loss(self, fake_features_batch):
        real_features_batch, real_labels_batch = next(iter(self.teacher_train_dataloader))
        rf_sgm = torch.sigmoid(real_features_batch)
        ff_sgm = torch.sigmoid(fake_features_batch)
        mmdl = MMD_loss()
        mmd_loss = mmdl(ff_sgm, rf_sgm)
        return mmd_loss

    def mean_std_loss(self):
        pass

    # Obtain data center for each class
    def obtain_rf_center(self, model):
        model.eval()
        features, labels = self.teacher_train_dataloader.dataset.tensors[0], \
                           self.teacher_train_dataloader.dataset.tensors[1]
        _, inter_output = model(features)
        unique_label = torch.unique(labels)
        center = torch.zeros(1, inter_output.size(1)).to(self.device)
        for i, j in enumerate(unique_label, 0):
            cls_id = torch.tensor([j], dtype=torch.int32).expand(labels.size(0), 1).to(args.cuda_device)
            # noinspection PyTypeChecker
            cls_idx = torch.nonzero((cls_id == labels) == 1)[:, 0]
            cls_center = inter_output[cls_idx].mean(0).view(1, -1)
            center = cls_center if i == 0 else torch.cat((center, cls_center))
        return center

    def obtain_real_data(self):
        teacher_indices = {'a': 'train_allclass_loc',
                           'b': 'test_allclass_loc'} if self.args.task_categories == 'GZSL_all' else {
            'a': 'trainval_loc', 'b': 'test_seen_loc'}
        need_realign_label = False if self.args.task_categories == 'GZSL_all' else True
        train_features, train_labels, val_features, val_labels = self.pro_data.cuda_data(
            self.pro_data.split_data_by_indices(**teacher_indices))
        batch_size = train_features.size(0) if self.args.loss_type == 'ct_loss' else self.args.batchTS
        train_data_loader = self.pro_data.create_dataloader(train_features, train_labels, batch_size,
                                                            need_realignment=need_realign_label)
        val_data_loader = self.pro_data.create_dataloader(val_features, val_labels, self.args.batchTS,
                                                          need_realignment=need_realign_label)
        return train_data_loader, val_data_loader


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class CenterLoss(nn.Module):
    def __init__(self, feat_dim=512):
        super(CenterLoss, self).__init__()
        self.target_center = torch.zeros(1, feat_dim).to(args.cuda_device)
        self.MSE_loss = nn.MSELoss().to(args.cuda_device)

    def forward(self, input_x, input_labels, target_x):
        for i, j in enumerate(input_labels, 0):
            self.target_center = target_x[j].view(1, -1) if i == 0 else torch.cat(
                (self.target_center, target_x[j].view(1, -1)))
        center_loss = self.MSE_loss(input_x, self.target_center)
        return center_loss


'''
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)  # (20,2048)-->(20,1)-->(20，10) +(10,2048)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
'''
