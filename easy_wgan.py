import argparse
import os
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vtls
import time
from model import *
from torch import autograd
torch.autograd.set_detect_anomaly(True)


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


def main(args):
    print(f'***\t start training \t***')
    # 参数设置
    batch_size = args.batch_size
    nz = args.nz
    ngf = args.ngf
    ndf = args.ndf
    lr = args.lr
    n_epochs = args.n_epochs
    image_size = args.image_size
    lambda_gp = args.lambda_gp

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    one = torch.tensor(1, dtype=torch.float).cuda()
    mone = (one * -1).cuda()
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 网络初始化
    net_g = Generator(ngf, nz).cuda()
    net_d = Discriminator(ndf).cuda()

    # 优化器
    epsilon = 1e-8
    beta1 = 0.5
    beta2 = 0.9
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr * 1.11,  betas=(beta1, beta2), eps=epsilon)
    d_loss = 0
    g_loss = 0
    fake_data = None
    st = time.time()

    # 训练循环
    for epoch in range(n_epochs):
        for p in net_d.parameters():
            p.requires_grad = True
        epost = time.time()
        for i, data in enumerate(dataloader, 0):
            net_d.zero_grad()
            real_data = data[0].cuda()
            # batch_size = real_data.size(0)

            # 判别器训练
            optimizer_d.zero_grad()
            noise = torch.randn(real_data.size(0), nz * 4).cuda()
            with torch.no_grad():
                noise_v = autograd.Variable(noise)
            fake_data = net_g(noise_v).detach()
            real_d_loss = net_d(real_data).mean()
            fake_d_loss = net_d(fake_data).mean()
            gp = gradient_penalty(net_d, real_data, fake_data)
            d_loss = - torch.mean(real_d_loss) + torch.mean(fake_d_loss) + gp * lambda_gp
            # Wasserstein_D = real_d_loss - fake_output
            d_loss.backward(retain_graph=True)
            optimizer_d.step()

            for p in net_d.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            for p in net_d.parameters():
                p.requires_grad = False

            # 生成器训练
            if i % args.n_critic == 0:
                net_g.zero_grad()
                optimizer_g.zero_grad()
                noise_g = torch.randn(real_data.size(0), nz * 4).cuda()
                noise_g_v = autograd.Variable(noise_g)
                fake_data_gen = net_g(noise_g_v)
                g_loss = net_d(fake_data_gen).mean()
                g_loss.backward(mone)
                g_loss = -g_loss
                optimizer_g.step()

            lambda_gp = adjust_lambda_gp(d_loss.item(), 5, lambda_gp)

        print(f"Epoch [{epoch + 1}/{n_epochs}]")
        print(f"Discriminator Loss: {d_loss.item()},\t Generator Loss: {g_loss.item()}")
        print(f"Epoch used time: {time.time() - epost}s")
        print("=" * 30 )
        # 可视化生成器进展
        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists('result/'):
                os.makedirs('result/')
            vtls.save_image(fake_data[:64].detach(), f'result/generated_image_epoch_{epoch + 1}.png', normalize=True)

        # 保存模型检查点
        if (epoch + 1) % args.save_model_interval == 0:
            if not os.path.exists('pth/'):
                os.makedirs('pth/')
            torch.save(net_g.state_dict(), f'pth/generator_model_epoch_{epoch + 1}.pt')
            torch.save(net_d.state_dict(), f'pth/discriminator_model_epoch_{epoch + 1}.pt')
            print(f"Nearly {args.save_model_interval} use time: {(time.time() - st)/3600}h")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='img', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--nz', type=int, default=96, help='Size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=96, help='Number of generator filters')
    parser.add_argument('--ndf', type=int, default=96, help='Number of discriminator filters')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=96, help='Size of generated images')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Weight of gradient penalty')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of critic iterations per generator iteration')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving generated images')
    parser.add_argument('--save_model_interval', type=int, default=200, help='Interval for saving model checkpoints')
    parser.add_argument('--clip_value', type=float, default=1, help='gradient clip')
    args = parser.parse_args()

    main(args)
