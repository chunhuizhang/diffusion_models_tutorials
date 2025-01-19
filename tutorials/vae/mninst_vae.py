# prerequisites
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def save_model(self, path):
        """保存模型参数和架构信息"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture': {
                'x_dim': self.fc1.in_features,
                'h_dim1': self.fc1.out_features,
                'h_dim2': self.fc2.out_features,
                'z_dim': self.fc31.out_features
            }
        }, path)
    
    @staticmethod
    def load_model(path):
        """加载保存的模型"""
        checkpoint = torch.load(path)
        architecture = checkpoint['architecture']
        model = VAE(
            x_dim=architecture['x_dim'],
            h_dim1=architecture['h_dim1'],
            h_dim2=architecture['h_dim2'],
            z_dim=architecture['z_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE, KLD


def train(epoch, train_history):
    vae.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        bce, kld = loss_function(recon_batch, data, mu, log_var)
        loss = bce + kld
        loss.backward()
        train_loss += bce.item() + kld.item()
        bce_loss += bce.item()
        kld_loss += kld.item()
        optimizer.step()
    
    # 计算平均损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_bce_loss = bce_loss / len(train_loader.dataset)
    avg_kld_loss = kld_loss / len(train_loader.dataset)
    
    # 记录历史
    train_history['total_loss'].append(avg_train_loss)
    train_history['bce_loss'].append(avg_bce_loss)
    train_history['kld_loss'].append(avg_kld_loss)
    
    print(f'====> Epoch: {epoch}, Average loss: {avg_train_loss:.4f}, '
          f'BCE loss: {avg_bce_loss:.4f}, '
          f'KLD loss: {avg_kld_loss:.4f}')


def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            # sum up batch loss
            bce, kld = loss_function(recon, data, mu, log_var)
            test_loss += bce.item() + kld.item()
        
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')


def visualize_latent_space(vae, data_loader, z_dim, num_batches=10):
    """可视化潜在空间的分布"""
    vae.eval()
    z_points = []
    labels = []
    
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.cuda()
            mu, _ = vae.encoder(data.view(-1, 784))
            z_points.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    z_points = np.concatenate(z_points, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 创建散点图
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], 
                         c=labels, cmap='tab10', 
                         alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('Latent Space Distribution')
    plt.savefig(f'./samples/latent_space_{z_dim}.png')
    plt.close()
    
    return z_points, labels


def plot_training_curves(train_history, z_dim):
    """绘制训练过程中的损失曲线"""
    plt.figure(figsize=(10, 6))
    
    # 在同一张图上绘制BCE和KLD损失
    plt.plot(train_history['bce_loss'], label='BCE Loss')
    plt.plot(train_history['kld_loss'], label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses (z_dim={z_dim})')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./figures/training_curves_{z_dim}.png')
    plt.close()


if __name__ == "__main__":

    bs = 2048
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # build model
    z_dim = 2
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=z_dim)
    vae.cuda()

    if os.path.exists(f'./models/vae_mnist_{z_dim}.pth'):
        vae = VAE.load_model(f'./models/vae_mnist_{z_dim}.pth')
        vae.cuda()
        test()
    else:
        # 初始化训练历史记录
        train_history = {
            'total_loss': [],
            'bce_loss': [],
            'kld_loss': []
        }
        
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        for epoch in tqdm(range(1, 51), desc="Training"):
            train(epoch, train_history)
        test()
        vae.save_model(f'./models/vae_mnist_{z_dim}.pth')
        
        # 绘制训练曲线
        plot_training_curves(train_history, z_dim=z_dim)  # 使用实际的z_dim值

    # 在训练完成后添加可视化
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
        
    # 可视化潜在空间
    z_points, labels = visualize_latent_space(vae, test_loader, z_dim)
    
    # 生成数字过渡图像
    with torch.no_grad():
        # 创建 16x16 的网格
        x = torch.linspace(-1.5, 1.5, 30)
        y = torch.linspace(-1.5, 1.5, 30)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        z = torch.stack([xx.flatten(), yy.flatten()], dim=1).cuda()
        
        # 生成图像
        sample = vae.decoder(z).cuda()
        save_image(sample.view(900, 1, 28, 28),
                  f'./samples/sample_grid_{z_dim}.png',
                  nrow=30,
                  normalize=True)

    with torch.no_grad():
        # 创建 30x30 的网格
        x = torch.linspace(-1, 1, 10)
        y = torch.linspace(-1, 1, 10)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        z = torch.stack([xx.flatten(), yy.flatten()], dim=1).cuda()
        
        # 生成图像
        sample = vae.decoder(z).cpu()
        
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                img = sample[idx].view(28, 28).numpy()
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(f'{x[j]:.1f}, {y[i]:.1f}')

        plt.tight_layout()
        plt.savefig(f'./samples/sample_grid_with_coords_{z_dim}.png', dpi=300, bbox_inches='tight')
        plt.close()

