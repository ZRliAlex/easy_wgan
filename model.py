import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.up(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return residual + out


class ResidualBlockWithDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlockWithDownSampling, self).__init__()

        self.down = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.down(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return residual + out


class Generator(nn.Module):
    def __init__(self, ngf, nz):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.nz = nz
        self.fc = nn.Linear(nz * 4, nz * 16 * 3 * 3)
        self.de_conv = nn.Sequential(
            # 增加更多的转置卷积层
            #nn.ConvTranspose2d(nz, 128, 3, 1, 0),  # 3
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.ConvTranspose2d(nz * 16, ngf * 8, 4, 2, 1),  # 6
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # 12
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # 24
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1),  # 48
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 1, 3, 4, 2, 1),  # 96
            # nn.BatchNorm2d(ngf * 1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(ngf, 3, 4, 2, 1),  # 输出3个通道，96x96
            nn.Tanh()
        )


    def forward(self, input_data):
        out = self.fc(input_data)
        out = out.view(out.size(0), self.nz * 16, 3, 3)
        out = self.de_conv(out)
        return out.view(-1, 3, 96, 96)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1 ),  # 输入是3通道图像，输出ndf通道，96x96 -> 48x48
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),  # 输出ndf * 2通道，48x48 -> 24x24
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),  # 输出ndf * 4通道，24x24 -> 12x12
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),  # 输出ndf * 8通道，12x12 -> 6x6
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),  # 输出ndf * 16通道，6x6 -> 3x3
            # nn.BatchNorm2d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf * 16, 1, 3, 1, 0, bias=False),  # 输出1通道，3x3 -> 1x1
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.lr = nn.Linear(ndf * 16 * 3 * 3, 1)

    def forward(self, input_data):
        out = self.main(input_data)
        out = out.view(-1, self.ndf * 16 * 3 * 3)
        out = self.lr(out)
        return out



def gradient_penalty(net_d, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    prob_interpolated = net_d(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradients_penalty


lambda_gp_min = 1
lambda_gp_max = 100.0


def adjust_lambda_gp(d_loss, threshold=0.001, lambda_gp=10):
    if abs(d_loss) < threshold:
        # 如果判别器损失较小，增加梯度惩罚权重
        lambda_gp_adjust = min(lambda_gp * 1.2, lambda_gp_max)
    elif (abs(d_loss) >= threshold) and (abs(d_loss) <= 2 * threshold):
        return lambda_gp
    else:
        # 如果判别器损失较大，减小梯度惩罚权重
        lambda_gp_adjust = max(lambda_gp / 1.2, lambda_gp_min)
    return lambda_gp_adjust
