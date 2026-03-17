import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from sklearn.decomposition import PCA

try:
    from openTSNE import TSNE
except ImportError:
    print("❌ 错误: 未找到 openTSNE 库。")
    print("请使用以下命令安装: pip install opentsne")
    sys.exit(1)


def load_data(file_path):
    """
    加载特征数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print(f"正在加载数据: {file_path} ...")
    data = np.load(file_path)

    features = data['features']
    labels = data['labels']

    # 尝试加载 metadata 如果存在（用于区分有标签/无标签）
    metadata_path = file_path.replace('all_features.npz', 'all_features_metadata.npz')
    is_labeled = None
    if os.path.exists(metadata_path):
        try:
            metadata = np.load(metadata_path)
            if 'is_labeled' in metadata:
                is_labeled = metadata['is_labeled']
                print(f"成功加载元数据: {metadata_path}")
        except Exception as e:
            print(f"加载元数据失败 (忽略): {e}")

    print(f"数据加载完成: 特征维度 {features.shape}, 标签数量 {labels.shape[0]}")
    return features, labels, is_labeled


def run_tsne(features, n_components=2, perplexity=30, learning_rate='auto', n_jobs=-1, random_state=42):
    """
    使用 openTSNE 运行 t-SNE
    """
    print(f"开始运行 t-SNE (n_components={n_components}, perplexity={perplexity})...")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=True
    )

    embedding = tsne.fit(features)
    print("t-SNE 降维完成")
    return embedding


def plot_tsne(embedding, labels, is_labeled=None, save_path='tsne_result.png', title='t-SNE Visualization'):
    """
    绘制 t-SNE 结果
    """
    print(f"正在绘制图像并保存到 {save_path} ...")

    # 获取类别数量
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 设置颜色映射
    cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20

    plt.figure(figsize=(12, 10))

    # 如果有 is_labeled 信息，我们可以用不同的标记形状区分
    if is_labeled is not None:
        # 绘制无标签数据 (通常数量多，用小点，透明度高)
        unlabeled_mask = is_labeled == 0
        if np.sum(unlabeled_mask) > 0:
            scatter1 = plt.scatter(
                embedding[unlabeled_mask, 0],
                embedding[unlabeled_mask, 1],
                c=labels[unlabeled_mask],
                cmap=cmap,
                s=10,
                alpha=0.4,
                marker='o',
                label='Unlabeled'
            )

        # 绘制有标签数据 (通常数量少，用大点，不透明，加黑边)
        labeled_mask = is_labeled == 1
        if np.sum(labeled_mask) > 0:
            scatter2 = plt.scatter(
                embedding[labeled_mask, 0],
                embedding[labeled_mask, 1],
                c=labels[labeled_mask],
                cmap=cmap,
                s=50,
                alpha=1.0,
                marker='*',
                edgecolors='k',
                linewidths=0.5,
                label='Labeled'
            )
    else:
        # 只有一种样式
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap=cmap,
            s=15,
            alpha=0.6
        )

    # 添加颜色条（显示类别）
    # 创建一个伪造的 mappable 用于 colorbar，确保包含所有类别
    norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ticks=unique_labels)
    cbar.set_label('Class ID')

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 如果区分了 labeled/unlabeled，添加图例
    if is_labeled is not None:
        # 我们手动创建图例句柄
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Unlabeled (Circle)', markersize=8,
                   alpha=0.6),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', label='Labeled (Star)', markersize=12,
                   markeredgecolor='k')
        ]
        plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图像保存成功: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='使用 openTSNE 绘制 t-SNE 图')
    parser.add_argument('--input', type=str, required=True,
                        help='输入的 .npz 特征文件路径 (例如: output/all_features.npz)')
    parser.add_argument('--output', type=str, default='tsne_result.png', help='输出图像路径')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity 参数')
    parser.add_argument('--n_jobs', type=int, default=-1, help='并行任务数 (-1 表示使用所有 CPU)')
    parser.add_argument('--limit', type=int, default=0, help='限制处理的样本数量 (0 表示不限制，用于测试)')
    parser.add_argument('--use_pca', action='store_true', help='是否先用 PCA 降维后再进行 t-SNE')
    parser.add_argument('--pca_dim', type=int, default=50, help='PCA 降维到的维度 (仅在 use_pca 时生效)')

    args = parser.parse_args()

    try:
        features, labels, is_labeled = load_data(args.input)

        if args.limit > 0 and args.limit < features.shape[0]:
            print(f"⚠️ 仅使用前 {args.limit} 个样本进行测试...")
            features = features[:args.limit]
            labels = labels[:args.limit]
            if is_labeled is not None:
                is_labeled = is_labeled[:args.limit]

        if args.use_pca:
            target_dim = min(args.pca_dim, features.shape[1])
            print(f"开始执行 PCA 降维: {features.shape[1]} → {target_dim}")
            pca = PCA(n_components=target_dim, random_state=42)
            features = pca.fit_transform(features).astype(np.float32)
            explained = float(np.sum(pca.explained_variance_ratio_))
            print(f"PCA 完成: 新特征维度 {features.shape}, 累积解释方差比 {explained:.4f}")

        embedding = run_tsne(
            features,
            perplexity=args.perplexity,
            n_jobs=args.n_jobs
        )

        plot_tsne(
            embedding,
            labels,
            is_labeled=is_labeled,
            save_path=args.output,
            title=f't-SNE Visualization (N={features.shape[0]}, Perplexity={args.perplexity})'
        )

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
