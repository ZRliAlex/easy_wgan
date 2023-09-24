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
