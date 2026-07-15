import random
import torch
import torchvision.transforms.functional as F


def crop(image, target, region):
    """
    元のDETR系cropロジック(そのまま流用)。
    region = (crop_top, crop_left, crop_height, crop_width)
    """
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


class LandmarkSquareCrop(object):
    """
    血清上端 or 下端の「基準線」(bboxを2分割する横線)を中心に、
    y軸方向にランダムオフセットを与えた正方形クロップを生成するクラス。

    前提:
    - top用/bottom用でCOCO jsonファイル自体が分割されており、
      各jsonには単一カテゴリ(category_id=1)のアノテーションのみが
      含まれている。そのためこのクラス内でのカテゴリ絞り込み・
      label_idのリマップは不要。
    - x軸方向はデータセット全体で容器位置が一定であるため、
      crop幅=画像幅(オフセットなし)とする。
    - y軸方向は基準線 ± max_offset の範囲でランダムにcrop中心をずらす。
    - crop領域が画像の上端/下端をはみ出す場合は、はみ出した側の端を
      画像端に固定する(paddingなし)。crop_size < image_height を前提に、
      上下同時にはみ出すことはないため、crop_topを有効範囲へclampする
      だけで対応可能。

    Args:
        crop_size (int): 正方形クロップの一辺(デフォルト256)
        max_offset (int): 基準線を中心にy方向へずらす最大オフセット(px)
        target_type (str): 'top' または 'bottom'(ログ・エラーメッセージ用。
            処理そのものはjson側で既に単一カテゴリに絞られているため、
            挙動に影響しない)
    """

    def __init__(self, crop_size=256, max_offset=64, target_type="top"):
        assert target_type in ("top", "bottom"), \
            f"target_type must be 'top' or 'bottom', got {target_type}"
        self.crop_size = crop_size
        self.max_offset = max_offset
        self.target_type = target_type

    def _get_basis_line_y(self, target):
        """bboxから基準線(bboxを2分割する横線)のy座標を算出"""
        boxes = target["boxes"]
        if boxes.shape[0] == 0:
            return None
        # 1画像1bbox想定。複数存在する場合は先頭を使用
        box = boxes[0]
        y1, y2 = box[1].item(), box[3].item()
        basis_y = (y1 + y2) / 2.0
        return basis_y

    def __call__(self, img, target):
        image_width, image_height = img.size

        basis_y = self._get_basis_line_y(target)
        if basis_y is None:
            raise ValueError(
                f"target_type='{self.target_type}' 用のアノテーションが"
                f"この画像に存在しません。"
            )

        # y方向にランダムオフセットを付与してcrop中心を決定
        offset = random.uniform(-self.max_offset, self.max_offset)
        crop_center_y = basis_y + offset
        crop_top = int(round(crop_center_y - self.crop_size / 2.0))

        # 画像端をはみ出す場合はクリップ(paddingなし)
        # crop_size < image_height の前提のもとでは、上端・下端の
        # はみ出しは同時に発生しないため、単純なclampのみで対応可能
        max_top = image_height - self.crop_size
        crop_top = max(0, min(crop_top, max_top))

        # x方向はcrop幅=画像幅なのでオフセットなし
        crop_left = int(round((image_width - self.crop_size) / 2.0))

        region = (crop_top, crop_left, self.crop_size, self.crop_size)
        return crop(img, target, region)

# top用データセット(top用jsonを読み込むDatasetに適用)
top_transform = Compose([
    LandmarkSquareCrop(crop_size=256, max_offset=64, target_type="top"),
    RandomHorizontalFlip(),   # 水平反転のみ。垂直反転は使用しない
    ToTensor(),
    Normalize(...),
])

# bottom用データセット(bottom用jsonを読み込むDatasetに適用)
bottom_transform = Compose([
    LandmarkSquareCrop(crop_size=256, max_offset=64, target_type="bottom"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(...),
])





import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

# 前段で定義済みの crop / LandmarkSquareCrop をそのまま import する想定
# from your_module import crop, LandmarkSquareCrop


class CocoAnnotationLoader:
    """
    top用 or bottom用(単一カテゴリ)のCOCO jsonを読み込み、
    image_id指定でPIL画像とtarget dict(boxes, labels, area, iscrowd)を返す。
    """

    def __init__(self, json_path, image_dir):
        self.image_dir = Path(image_dir)
        with open(json_path, "r") as f:
            coco = json.load(f)

        self.images_by_id = {img["id"]: img for img in coco["images"]}

        self.anns_by_image_id = {}
        for ann in coco["annotations"]:
            self.anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    def get_image_ids(self):
        return list(self.images_by_id.keys())

    def load(self, image_id):
        img_info = self.images_by_id[image_id]
        img_path = self.image_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self.anns_by_image_id.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            boxes.append([x, y, x + w, y + h])  # -> [x1, y1, x2, y2]
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([img.height, img.width]),
            "size": torch.tensor([img.height, img.width]),
        }
        return img, target


def get_basis_line_y_from_target(target):
    """targetのboxesから基準線(bboxを2分割する横線)のy座標を返す。複数あれば全て返す"""
    ys = []
    for box in target["boxes"]:
        y1, y2 = box[1].item(), box[3].item()
        ys.append((y1 + y2) / 2.0)
    return ys


def visualize_crop(loader, image_id, target_type="top",
                    crop_size=256, max_offset=64, n_samples=3, seed=None):
    """
    指定したimage_idについて、元画像+基準線 と
    n_samples回分のクロップ結果+基準線 を並べて表示する。
    """
    if seed is not None:
        random.seed(seed)

    img, target = loader.load(image_id)
    image_width, image_height = img.size

    # 元画像上の基準線(全カテゴリ分。デモとして色分けして描画)
    basis_ys_all = get_basis_line_y_from_target(target)
    labels_all = target["labels"].tolist()

    cropper = LandmarkSquareCrop(
        crop_size=crop_size, max_offset=max_offset, target_type=target_type
    )

    fig, axes = plt.subplots(1, n_samples + 1, figsize=(4 * (n_samples + 1), 5))

    # --- 元画像の表示 ---
    ax0 = axes[0]
    ax0.imshow(img)
    ax0.set_title(f"Original image (id={image_id})\n{image_width}x{image_height}")
    for y, label in zip(basis_ys_all, labels_all):
        color = "red" if label == 1 else "blue"  # jsonがtop単独/bottom単独なら通常1色
        ax0.axhline(y=y, color=color, linewidth=1.5)
    ax0.set_xlim(0, image_width)
    ax0.set_ylim(image_height, 0)  # 画像座標系(上が0)に合わせる

    # --- クロップ結果を n_samples 回生成して表示 ---
    for i in range(n_samples):
        cropped_img, cropped_target = cropper(img, target)

        ax = axes[i + 1]
        ax.imshow(cropped_img)

        basis_ys_cropped = get_basis_line_y_from_target(cropped_target)
        cropped_labels = cropped_target["labels"].tolist()

        for y, label in zip(basis_ys_cropped, cropped_labels):
            color = "red" if label == 1 else "blue"
            ax.axhline(y=y, color=color, linewidth=2)

        ax.set_title(f"Cropped sample {i+1}\n{crop_size}x{crop_size}")
        ax.set_xlim(0, crop_size)
        ax.set_ylim(crop_size, 0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ------- 設定(適宜変更してください) -------
    JSON_PATH = "/path/to/top_annotations.json"   # top用 or bottom用のCOCO json
    IMAGE_DIR = "/path/to/images"                 # 画像ディレクトリ
    TARGET_TYPE = "top"                           # "top" or "bottom"
    CROP_SIZE = 256
    MAX_OFFSET = 64
    N_SAMPLES = 4                                 # 1画像あたり何パターンのクロップを見るか
    SEED = 0
    # -----------------------------------------

    loader = CocoAnnotationLoader(JSON_PATH, IMAGE_DIR)

    image_ids = loader.get_image_ids()
    # ランダムに何枚かサンプル表示する例
    sample_ids = random.sample(image_ids, k=min(3, len(image_ids)))

    for image_id in sample_ids:
        visualize_crop(
            loader,
            image_id=image_id,
            target_type=TARGET_TYPE,
            crop_size=CROP_SIZE,
            max_offset=MAX_OFFSET,
            n_samples=N_SAMPLES,
            seed=SEED,
        )