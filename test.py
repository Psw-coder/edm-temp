from training.networks import SongUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import math
import random
import numpy as np

# Diffusion model related

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)






class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10, task='fashion'):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.time_dim = 64

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # self.timeembed1 = EmbedFC(1, 2 * n_feat)
        # self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.timeembed1 = EmbedFC(self.time_dim, 2 * n_feat)
        self.timeembed2 = EmbedFC(self.time_dim, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        ks = 7 if task in ['fashion', 'mnist'] else 8
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, ks, ks),  # otherwise just have 2*n_feat
            # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),  # otherwise just have 2*n_feat
            # TODO: for cifar10, it should be set to (2 * n_feat, 2 * n_feat, 8, 8)
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    # def forward(self, x, c, t, context_mask):
    def forward(self, x, t, c):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # embed context, time step
        t = timestep_embedding(t, self.time_dim)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion * planes)
                nn.GroupNorm(32, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.GroupNorm(32, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(data_root: str, batch_size: int, num_workers: int = 4):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(model, optimizer, epoch, acc, save_dir: Path, tag: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"epoch{epoch:03d}_{tag}_acc{acc*100:.2f}.pth"
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'acc': acc,
    }, ckpt_path)
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10 with interval checkpointing')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/cifar10')
    parser.add_argument('--data_root', type=str, default='./samples/cifar10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=6)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_loader, test_loader = get_cifar10_loaders(args.data_root, args.batch_size, args.num_workers)

    # Model, loss, optimizer, scheduler
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr_cur = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{args.epochs} | LR {lr_cur:.4f} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Test Loss {test_loss:.4f} Acc {test_acc*100:.2f}%")

        # Save at interval
        if epoch % args.save_interval == 0:
            ckpt_path = save_checkpoint(model, optimizer, epoch, test_acc, save_dir, tag='interval')
            print(f"Saved interval checkpoint: {ckpt_path}")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = save_checkpoint(model, optimizer, epoch, test_acc, save_dir, tag='best')
            print(f"Saved best checkpoint: {ckpt_path}")

    final_path = save_checkpoint(model, optimizer, args.epochs, test_acc, save_dir, tag='final')
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} min. Final checkpoint: {final_path}")


if __name__ == '__main__':
    from thop import profile
    from tools.flops import compute_model_flops_and_params
    from ACGAN_training.ACGAN import ACGAN
    device = "cuda:0"
    model = SongUNet(img_resolution=32, in_channels=3, out_channels=3,
                     label_dim=0,  # Number of class labels, 0 = unconditional.
                     augment_dim=9,  # Augmentation label dimensionality, 0 = no augmentation.

                     model_channels=48,  # Base multiplier for the number of channels.
                     channel_mult=[1, 1, 1],  # Per-resolution multipliers for the number of channels.
                     channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
                     num_blocks=4,  # Number of residual blocks per resolution.
                     attn_resolutions=[16],  # List of resolutions with self-attention.
                     dropout=0.13,  # Dropout probability of intermediate activations.
                     label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.

                     embedding_type='positional',
                     # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                     channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                     encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                     decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                     resample_filter=[1, 1]).to(device)

    x = torch.randn(1, 3, 32, 32, device=device)
    sigma = torch.tensor([0.5], device=device, dtype=torch.float32)
    flops, params_mb = compute_model_flops_and_params(model, (x, sigma, sigma))
    print(f"Edm       :{flops / 1e9:.4f} G FLOPs")
    print(f"Params:{sum([p.numel() for p in model.parameters()])}")

    x = torch.FloatTensor(1, 3, 32, 32)
    c = torch.FloatTensor(1, 10)
    t = torch.FloatTensor(1)
    N = ContextUnet(3, 128, 10, task='cifar10')
    macs, params = profile(N, inputs=(x, t, c))
    # print(macs)
    print(f"ContextUnet:{macs/1e9:.4f} G FLOPs")
    print(f"Paramas:{sum([p.numel() for p in N.parameters()])}")

    # gan
    gan = ACGAN(img_dim=3, label_dim=10, img_size=32,device='cpu')
    c = torch.zeros((1,10),dtype=torch.long)
    # macs, params = compute_model_flops_and_params(gan, (x,c))
    # print(f"GAN   : {macs / 1e9:.4f} G FLOPs")
    print(f"Gan Params:{sum([p.numel() for p in gan.parameters()])}")

    #resnet
    x = torch.FloatTensor(1,1, 3, 32, 32)
    N = ResNet18(10)
    # macs, params = profile(N, inputs=x)
    macs, params = compute_model_flops_and_params(N, (x))
    print(f"Resnet    : {macs/1e9:.4f} G FLOPs")
    print(f"Params:{sum([p.numel() for p in N.parameters()])}")


    # Classifier = ResNet18(10)
    # macs, params = get_model_complexity_info(Classifier, input_res=(1,), input_constructor=prepare_input_cifar10, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # # print('Params and FLOPs are {}M and {}G'.format(params/1e6, flops/1e9))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # macs, params = get_model_complexity_info(N, input_res=(1,), input_constructor=prepare_input, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('Params and FLOPs are {}M and {}G'.format(params/1e6, flops/1e9))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # main()