import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import pickle
import tarfile
from PIL import Image
import click
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
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
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
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


# ----------------------------------------------------------------------------
# 自定义数据集类：加载保存的伪标签图像

class PseudoLabelDataset(Dataset):
    """
    加载通过伪标签保存的图像数据集
    """

    def __init__(self, pseudo_label_dir, transform=None, include_discarded=False):
        """
        Args:
            pseudo_label_dir: 伪标签图像保存的根目录 (包含 pseudo_labeled_images 文件夹)
            transform: 图像变换
            include_discarded: 是否包含被丢弃的样本
        """
        self.pseudo_label_dir = pseudo_label_dir
        self.transform = transform
        self.include_discarded = include_discarded

        # CIFAR-10类别名称
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        self.images = []
        self.labels = []
        self.image_paths = []

        self._load_images()

    def _load_images(self):
        """加载所有图像路径和标签"""
        images_dir = os.path.join(self.pseudo_label_dir, 'pseudo_labeled_images')

        if not os.path.exists(images_dir):
            raise ValueError(f"找不到伪标签图像目录: {images_dir}")

        print(f"从 {images_dir} 加载伪标签数据集...")

        # 加载各个类别的图像
        for class_id in range(10):
            class_dir = os.path.join(images_dir, f'class_{class_id}_{self.class_names[class_id]}')
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.png'):
                        image_path = os.path.join(class_dir, filename)
                        self.image_paths.append(image_path)
                        self.labels.append(class_id)

        # 如果包含被丢弃的样本，将其标签设为-1（或者跳过）
        if self.include_discarded:
            discarded_dir = os.path.join(images_dir, 'discarded')
            if os.path.exists(discarded_dir):
                for filename in os.listdir(discarded_dir):
                    if filename.endswith('.png'):
                        # 从文件名中提取真实标签
                        # 格式: discarded_0001_true_7_frog.png
                        parts = filename.split('_')
                        if len(parts) >= 4 and parts[2] == 'true':
                            true_label = int(parts[3])
                            image_path = os.path.join(discarded_dir, filename)
                            self.image_paths.append(image_path)
                            self.labels.append(true_label)  # 使用真实标签

        print(f"加载了 {len(self.image_paths)} 张图像")

        # 统计各类别数量
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print("各类别样本数量:")
        for class_id in range(10):
            count = label_counts.get(class_id, 0)
            print(f"  类别 {class_id} ({self.class_names[class_id]}): {count} 张")

        if self.include_discarded and -1 in label_counts:
            print(f"  被丢弃样本: {label_counts[-1]} 张")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------------------------------------------
# CIFAR-10测试数据集加载器

class CIFAR10TestDataset(Dataset):
    """
    加载CIFAR-10测试数据集
    """

    def __init__(self, tarball_path, transform=None):
        self.tarball_path = tarball_path
        self.transform = transform

        # 加载测试数据
        self.images, self.labels = self._load_test_data()

    def _load_test_data(self):
        """从CIFAR-10 tar.gz文件加载测试数据"""
        print(f"从 {self.tarball_path} 加载CIFAR-10测试数据...")

        with tarfile.open(self.tarball_path, 'r:gz') as tar:
            # 加载测试批次
            member = tar.getmember('cifar-10-batches-py/test_batch')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')

            images = data['data'].reshape(-1, 3, 32, 32)
            labels = np.array(data['labels'])

            # 转换为HWC格式
            images = images.transpose(0, 2, 3, 1)

        print(f"加载了 {len(images)} 张测试图像")
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换为PIL图像
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------------------------------------------------------
# 训练函数

def train_classifier(model, train_loader, test_loader, device, num_epochs=100,
                     learning_rate=0.001, save_dir='classifier_checkpoints'):
    """
    训练分类器
    """
    os.makedirs(save_dir, exist_ok=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_acc = 0.0

    print(f"\n开始训练分类器...")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"设备: {device}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {num_epochs}")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

            # 更新进度条
            train_acc = 100. * correct_train / total_train
            train_pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{train_acc:.2f}%'
            })

        # 计算训练准确率和损失
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train

        # 测试阶段
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()

                test_acc = 100. * correct_test / total_test
                test_pbar.set_postfix({'Acc': f'{test_acc:.2f}%'})

        epoch_test_acc = 100. * correct_test / total_test

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        test_accuracies.append(epoch_test_acc)

        # 打印epoch结果
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Test Acc: {epoch_test_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # 保存最佳模型
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': epoch_test_acc,
                'train_acc': epoch_train_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'  ✅ 保存最佳模型 (测试准确率: {best_test_acc:.2f}%)')

        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': epoch_test_acc,
                'train_acc': epoch_train_acc,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        print('-' * 60)

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_test_acc': best_test_acc
    }

    # 绘制训练曲线
    plot_training_curves(history, save_dir)

    print(f"\n🎉 训练完成！")
    print(f"最佳测试准确率: {best_test_acc:.2f}%")
    print(f"模型和历史保存在: {save_dir}")

    return history


def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['test_accuracies'], label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线保存到: {os.path.join(save_dir, 'training_curves.png')}")


# ----------------------------------------------------------------------------
# 主函数

@click.command()
@click.option('--pseudo_label_dir', help='伪标签图像保存的目录', metavar='PATH', type=str, required=True)
@click.option('--cifar10_data', help='CIFAR-10 tar.gz文件路径（用于测试集）', metavar='PATH', type=str, required=True)
@click.option('--save_dir', help='模型保存目录', metavar='DIR', type=str, default='classifier_checkpoints')
@click.option('--batch_size', help='批处理大小', metavar='INT', type=int, default=128)
@click.option('--num_epochs', help='训练轮数', metavar='INT', type=int, default=100)
@click.option('--learning_rate', help='学习率', metavar='FLOAT', type=float, default=0.001)
@click.option('--include_discarded', help='是否包含被丢弃的样本', is_flag=True, default=False)
@click.option('--num_workers', help='数据加载器工作进程数', metavar='INT', type=int, default=4)
def main(pseudo_label_dir, cifar10_data, save_dir, batch_size, num_epochs,
         learning_rate, include_discarded, num_workers):
    """
    使用伪标签数据训练CIFAR-10分类器
    """
    print("=" * 60)
    print("🚀 CIFAR-10 伪标签分类器训练")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 创建数据集
    print("\n📁 加载数据集...")
    train_dataset = PseudoLabelDataset(
        pseudo_label_dir=pseudo_label_dir,
        transform=train_transform,
        include_discarded=include_discarded
    )

    test_dataset = CIFAR10TestDataset(
        tarball_path=cifar10_data,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建模型
    print("\n🏗️ 创建模型...")
    model = ResNet18(num_classes=10).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 开始训练
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=save_dir
    )

    print("\n📊 训练总结:")
    print(f"最终训练准确率: {history['train_accuracies'][-1]:.2f}%")
    print(f"最终测试准确率: {history['test_accuracies'][-1]:.2f}%")
    print(f"最佳测试准确率: {history['best_test_acc']:.2f}%")


if __name__ == "__main__":
    main()