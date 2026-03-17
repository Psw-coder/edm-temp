#!/bin/bash

# pkl 文件所在的目录
SEARCH_DIR="/data/psw/edm/run/00008-cifar10-32x32-uncond-ddpmpp-edm-gpus4-batch512-fp32"
BASE_OUTDIR="pseudo_label_output_uncond"
INTERVAL=2  # 设置间隔：每隔多少个文件处理一次（1表示处理所有文件，2表示每隔1个处理1个，以此类推）

echo "删除遗留位置的特征$BASE_OUTDIR"
# 注意：这里原来删除的是 pseudo_label_output_uncond，建议确认是否要删除 BASE_OUTDIR
# 如果目的是清理输出目录，应该删除 $BASE_OUTDIR
if [ -d "$BASE_OUTDIR" ]; then
    echo "清理输出目录: $BASE_OUTDIR"
    rm -r "$BASE_OUTDIR"
fi

# 检查目录是否存在
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR does not exist."
    exit 1
fi

echo "开始遍历目录: $SEARCH_DIR"
echo "处理间隔: $INTERVAL"

# 初始化计数器
count=0

# 遍历所有符合模式的 pkl 文件
# 使用 sort 确保文件按文件名排序
for pklpath in $(ls "$SEARCH_DIR"/network-snapshot-*.pkl | sort); do
    # 检查文件是否存在（防止 glob 不匹配时的错误）
    [ -e "$pklpath" ] || continue

    # 检查是否满足间隔要求
    if (( count % INTERVAL != 0 )); then
        ((count++))
        continue
    fi
    ((count++))

    # 提取文件名 (例如: network-snapshot-040000.pkl)
    filename=$(basename "$pklpath")

    # 提取序号 (例如: 040000)
    # 假设格式固定为 network-snapshot-数字.pkl
    seq_num=$(echo "$filename" | sed -E 's/network-snapshot-([0-9]+)\.pkl/\1/')

    # 定义该文件的输出目录
    outdir="${BASE_OUTDIR}/${seq_num}"

    echo "--------------------------------------------------"
    echo "正在处理文件: $filename (索引: $((count-1)))"
    echo "序号: $seq_num"
    echo "输出目录: $outdir"

    # 如果输出子目录已存在，则清理它
    if [ -d "$outdir" ]; then
        echo "清理已存在的输出目录: $outdir"
        rm -r "$outdir"
    fi

    # 执行 Python 脚本
    python pseudo_label.py --network_pkl "$pklpath" \
        --save_features \
        --outdir "$outdir" \
        --max_images 10000

    echo "处理完成: $seq_num"
done

echo "=================================================="
echo "所有任务执行完毕。"
