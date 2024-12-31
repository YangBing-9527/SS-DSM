import torch
import numpy as np
from torch import nn
from torch.autograd import grad


def seed_torch(seed=2021):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


class ACLoss(nn.Module):
    """
    Active Contour Loss
    based on sobel filter
    """

    def __init__(self, miu=1.0, classes=3):
        super(ACLoss, self).__init__()

        self.miu = miu
        self.classes = classes
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.sobel_x = nn.Parameter(torch.from_numpy(sobel_x).float().expand(self.classes, 1, 3, 3),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.from_numpy(sobel_y).float().expand(self.classes, 1, 3, 3),
                                    requires_grad=False)

        self.diff_x = nn.Conv2d(self.classes, self.classes, groups=self.classes,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.diff_x.weight.data = self.sobel_x
        self.diff_y = nn.Conv2d(self.classes, self.classes, groups=self.classes,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.diff_y.weight.data = self.sobel_y

    def forward(self, predication, label):
        grd_x = self.diff_x(predication)
        grd_y = self.diff_y(predication)

        predication.register_hook(save_grad('pred'))

        # length
        length = torch.sum(torch.abs(torch.sqrt(grd_x ** 2 + grd_y ** 2 + 1e-8)))
        # length = (length - length.min()) / (length.max() - length.min() + 1e-8)
        # length = torch.sum(length)

        # # region
        # label = label.float()
        # c_in = torch.ones_like(predication)
        # c_out = torch.zeros_like(predication)
        # region_in = torch.abs(torch.sum(predication * ((label - c_in) ** 2)))
        # region_out = torch.abs(torch.sum((1 - predication) * ((label - c_out) ** 2)))
        # region = self.miu * region_in + region_out

        return length
        # return region + length


# data = torch.randn((1, 1, 4, 4), requires_grad=True)
# data = torch.tensor([[[[0, 0.4, 0.5, 0], [0, 0.9, 0.8, 0], [0, 0.8, 0.7, 0],  [0, 0.4, 0.1, 0]]]], requires_grad=True)
# data = torch.tensor([[[[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0],  [0, 1., 1, 0]]]], requires_grad=True)
data = torch.tensor([[[[0.5, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0],  [0, 0.5, 0.5, 0]]]], requires_grad=True)
label = torch.Tensor([[[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0],  [0, 0, 0, 0]]]])
# label = torch.Tensor([[[[0, 0], [1, 1], [1, 1],  [0, 0]]]])
# label = torch.Tensor([[[[1, 1], [1, 1]]]])
Loss = ACLoss(classes=1)
# optimizer = torch.optim.Adam()


# output = torch.nn.MaxPool2d(kernel_size=2)(data)
print(data)
# print(output)
print(label)
# print(output.shape)
# print(label.shape)
# for epoch in range(5):

loss = Loss(data, label)
mse_loss1 = torch.nn.MSELoss()(data, label)
print(loss)

loss.backward()
print('data.grad:\n', data.grad)
# print('pred_grad:\n', grads['pred'])
data = data - 0.1*data.grad
print('new_data:\n', data)
mse_loss2 = torch.nn.MSELoss()(data, label)
print('pre_loss:', mse_loss1, 'after_loss:', mse_loss2)
