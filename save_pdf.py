import os
import math
import click
from PIL import Image
import generate as gen


def get_image_exts():
    Image.init()
    exts = set(Image.EXTENSION.keys())
    if len(exts) == 0:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    return exts


def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in get_image_exts()


def list_images(input_dir):
    try:
        entries = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    files = [os.path.join(input_dir, f) for f in entries]
    images = [p for p in files if os.path.isfile(p) and is_image_file(p)]
    return sorted(images)


def list_client_dirs(input_dir, client_ids):
    if client_ids is not None:
        dirs = [os.path.join(input_dir, str(cid)) for cid in client_ids]
        return [d for d in dirs if os.path.isdir(d)]
    images_in_root = list_images(input_dir)
    if images_in_root:
        return [input_dir]
    try:
        names = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    subdirs = [os.path.join(input_dir, d) for d in names]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    return [d for d in subdirs if list_images(d)]


def resize_cover(img, target_w, target_h):
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))


def resize_contain(img, target_w, target_h):
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(math.floor(w * scale))
    new_h = int(math.floor(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (target_w, target_h), color=(255, 255, 255))
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(img, (left, top))
    return canvas


def build_pages(image_paths, rows, cols, gap, page_width, page_height, fit_mode):
    cell_w = int(round((page_width - gap * (cols - 1)) / cols))
    cell_h = int(round((page_height - gap * (rows - 1)) / rows))
    cell_w = max(cell_w, 1)
    cell_h = max(cell_h, 1)
    total_w = cell_w * cols + gap * (cols - 1)
    total_h = cell_h * rows + gap * (rows - 1)
    offset_x = max((page_width - total_w) // 2, 0)
    offset_y = max((page_height - total_h) // 2, 0)
    per_page = rows * cols
    pages = []
    total_pages = (len(image_paths) + per_page - 1) // per_page
    for page_idx in range(total_pages):
        page = Image.new('RGB', (page_width, page_height), color=(255, 255, 255))
        start = page_idx * per_page
        end = min(start + per_page, len(image_paths))
        for i, img_path in enumerate(image_paths[start:end]):
            r = i // cols
            c = i % cols
            x = offset_x + c * (cell_w + gap)
            y = offset_y + r * (cell_h + gap)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if fit_mode == 'stretch':
                img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            elif fit_mode == 'contain':
                img = resize_contain(img, cell_w, cell_h)
            else:
                img = resize_cover(img, cell_w, cell_h)
            page.paste(img, (x, y))
        pages.append(page)
    return pages


def cm_to_px(cm, dpi):
    return int(round(cm / 2.54 * dpi))


'''
python save_pdf.py --input_dir /data/psw/edm/gen_out   --outdir pdf_out   --rows 8 --cols 6 --gap 4   --page_width_cm 20 --page_height_cm 28   --dpi 300

'''
@click.command()
@click.option('--input_dir', help='输入图片根目录', metavar='DIR', type=str, required=True)
@click.option('--outdir', help='输出 PDF 根目录', metavar='DIR', type=str, required=True)
@click.option('--rows', help='每页行数', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--cols', help='每页列数', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--gap', help='图片间距(像素)', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--page_width_cm', help='页面宽度(厘米)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), required=True)
@click.option('--page_height_cm', help='页面高度(厘米)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), required=True)
@click.option('--dpi', help='厘米转换像素的 DPI', metavar='INT', type=click.IntRange(min=1), default=300, show_default=True)
@click.option('--fit_mode', help='图像适配方式', metavar='cover|contain|stretch', type=click.Choice(['cover', 'contain', 'stretch']), default='cover', show_default=True)
@click.option('--client_ids', help='客户端 ID 列表 (e.g. 0-9)', metavar='LIST', type=gen.parse_int_list, default=None)
@click.option('--max_images', help='每个客户端最多拼接多少张', metavar='INT', type=click.IntRange(min=1), default=None)
def main(input_dir, outdir, rows, cols, gap, page_width_cm, page_height_cm, dpi, fit_mode, client_ids, max_images):
    input_dir = os.path.abspath(input_dir)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    client_dirs = list_client_dirs(input_dir, client_ids)
    if len(client_dirs) == 0:
        raise FileNotFoundError('未找到包含图像的客户端目录')

    page_width = cm_to_px(page_width_cm, dpi)
    page_height = cm_to_px(page_height_cm, dpi)

    for client_dir in client_dirs:
        image_paths = list_images(client_dir)
        if max_images is not None:
            image_paths = image_paths[:max_images]
        if len(image_paths) == 0:
            print(f'跳过空目录: {client_dir}')
            continue
        pages = build_pages(image_paths, rows, cols, gap, page_width, page_height, fit_mode)
        client_name = os.path.basename(client_dir.rstrip(os.sep))
        if client_dir == input_dir:
            client_name = 'root'
        pdf_path = os.path.join(outdir, f'Client-{client_name}.pdf')
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f'已保存: {pdf_path} (页数: {len(pages)}, 图片数: {len(image_paths)})')


if __name__ == '__main__':
    main()
