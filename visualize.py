"""
RandomWhiteLabelPaste デモスクリプト（実画像 + COCO JSON版）
=============================================================
256×1536 の実検体容器画像と COCO format アノテーションを読み込み、
データ拡張の効果を可視化する。
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw


# ─────────────────────────────────────────
#  設定
# ─────────────────────────────────────────

IMAGE_PATH  = "your_image.jpg"          # ← 対象画像のパスに変更
COCO_PATH   = "annotations.json"        # ← COCO JSON のパスに変更
OUTPUT_PATH = "augmentation_demo.png"
RANDOM_SEED = 0

CONFIGS = [
    dict(label="標準\n(1–3枚, 幅40–95%)",
         kwargs=dict(num_labels=(1, 3), label_width_ratio=(0.40, 0.95),
                     label_height_ratio=(0.15, 0.45), rotation_range=(-3, 3),
                     p=1.0, min_bbox_visibility=0.0)),
    dict(label="大きめ\n(幅80–100%)",
         kwargs=dict(num_labels=(1, 2), label_width_ratio=(0.80, 1.00),
                     label_height_ratio=(0.20, 0.50), rotation_range=(-5, 5),
                     p=1.0, min_bbox_visibility=0.0)),
    dict(label="中心線保護\n(visibility=1.0)",
         kwargs=dict(num_labels=(1, 3), label_width_ratio=(0.40, 0.95),
                     label_height_ratio=(0.15, 0.45), rotation_range=(-3, 3),
                     p=1.0, min_bbox_visibility=1.0)),
    dict(label="多数枚\n(3–5枚)",
         kwargs=dict(num_labels=(3, 5), label_width_ratio=(0.35, 0.70),
                     label_height_ratio=(0.10, 0.30), rotation_range=(-4, 4),
                     p=1.0, min_bbox_visibility=0.0)),
]

N_SAMPLES = 3   # 同一設定で何パターン生成して並べるか


# ─────────────────────────────────────────
#  COCO JSON パーサー
# ─────────────────────────────────────────

def load_coco_target(image_path: str, coco_path: str) -> dict:
    """
    COCO format JSON から指定画像に対応するアノテーションを読み込む。

    COCO bbox は [x, y, width, height]（xywh）形式で格納されているため、
    モデル入力に合わせて [x1, y1, x2, y2]（xyxy）形式に変換して返す。

    Returns:
        target (dict):
            boxes  : FloatTensor (N, 4)  — xyxy 形式
            labels : LongTensor  (N,)    — category_id
            area   : FloatTensor (N,)
            image_id : int
            category_id_to_name : dict   — 描画用ラベル名
    """
    image_path = Path(image_path)
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # file_name で画像エントリを検索（パスの末尾ファイル名で照合）
    image_entry = None
    for img in coco["images"]:
        if Path(img["file_name"]).name == image_path.name:
            image_entry = img
            break
    if image_entry is None:
        raise ValueError(
            f"'{image_path.name}' が COCO JSON の images に見つかりません。"
        )

    image_id = image_entry["id"]

    # category_id → name のマッピング
    cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    # 対象画像のアノテーションを収集
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    if not anns:
        raise ValueError(
            f"image_id={image_id} に対応するアノテーションが見つかりません。"
        )

    boxes, labels, areas = [], [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]           # COCO: xywh
        boxes.append([x, y, x + w, y + h]) # → xyxy
        labels.append(ann["category_id"])
        areas.append(ann.get("area", w * h))

    target = {
        "boxes":   torch.tensor(boxes,  dtype=torch.float32),
        "labels":  torch.tensor(labels, dtype=torch.long),
        "area":    torch.tensor(areas,  dtype=torch.float32),
        "image_id": image_id,
        "category_id_to_name": cat_map,
    }
    return target


# ─────────────────────────────────────────
#  RandomWhiteLabelPaste クラス
# ─────────────────────────────────────────

class RandomWhiteLabelPaste(object):
    def __init__(
        self,
        num_labels=(1, 3),
        label_width_ratio=(0.4, 0.95),
        label_height_ratio=(0.15, 0.45),
        fill_color=(255, 255, 255),
        border_color=(200, 200, 200),
        border_width=1,
        rotation_range=(-3, 3),
        p=0.5,
        min_bbox_visibility=0.0,
        white_pixel_thresh=240,   # 追加：白とみなすRGB各チャンネルの下限値
        max_white_ratio=0.5,      # 追加：bbox内白色ピクセル割合の上限（超えたらスキップ）
    ):
        self.num_labels = num_labels
        self.label_width_ratio = label_width_ratio
        self.label_height_ratio = label_height_ratio
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = border_width
        self.rotation_range = rotation_range
        self.p = p
        self.min_bbox_visibility = min_bbox_visibility
        self.white_pixel_thresh = white_pixel_thresh
        self.max_white_ratio = max_white_ratio 

    def _get_bbox_center_lines(self, target, img_h):
        center_ys = []
        if target is not None and "boxes" in target:
            for box in target["boxes"]:
                y_min, y_max = box[1].item(), box[3].item()
                center_ys.append((y_min + y_max) / 2.0)
        return center_ys

    def _label_covers_center_line(self, lx, ly, lw, lh, angle_deg, center_ys):
        if not center_ys:
            return False
        return any(ly <= cy <= ly + lh for cy in center_ys)

    def _generate_label_patch(self, lw, lh, angle_deg):
        patch = Image.new("RGBA", (lw, lh), (*self.fill_color, 255))
        draw = ImageDraw.Draw(patch)
        if self.border_color is not None:
            draw.rectangle(
                [0, 0, lw - 1, lh - 1],
                outline=(*self.border_color, 255),
                width=self.border_width,
            )
        num_lines = random.randint(1, max(1, lh // 14))
        for _ in range(num_lines):
            y_line = random.randint(6, max(6, lh - 6))
            x_start = random.randint(4, max(4, int(lw * 0.05)))
            x_end = min(int(lw * random.uniform(0.3, 0.85)) + x_start, lw - 4)
            if x_end > x_start:
                draw.line([(x_start, y_line), (x_end, y_line)],
                          fill=(180, 180, 180, 120), width=1)
        if angle_deg != 0:
            patch = patch.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        return patch

    def __call__(self, img, target=None):
        if random.random() > self.p:
            return img, target
        
        if target is not None and "boxes" in target:
        img_np = np.array(img)  # (H, W, 3)
        for box in target["boxes"]:
            x1, y1, x2, y2 = (int(v.item()) for v in box)
            crop = img_np[y1:y2, x1:x2]  # bbox領域を切り出し
            if crop.size == 0:
                continue
            # RGB全チャンネルが閾値以上のピクセルを白とみなす
            white_mask = np.all(crop >= self.white_pixel_thresh, axis=2)
            white_ratio = white_mask.sum() / (crop.shape[0] * crop.shape[1])
            if white_ratio > self.max_white_ratio:
                return img, target  # いずれかのbboxが閾値超えならスキップ

        img_w, img_h = img.size
        img_rgba = img.convert("RGBA")
        center_ys = self._get_bbox_center_lines(target, img_h)
        n_labels = random.randint(self.num_labels[0], self.num_labels[1])
        for _ in range(n_labels):
            lw = max(int(img_w * random.uniform(*self.label_width_ratio)), 10)
            lh = max(int(img_h * random.uniform(*self.label_height_ratio)), 8)
            angle_deg = random.uniform(*self.rotation_range)
            lx = random.randint(-lw // 4, max(0, img_w - lw * 3 // 4))
            ly = random.randint(0, max(0, img_h - lh))
            if self.min_bbox_visibility > 0.0:
                if self._label_covers_center_line(lx, ly, lw, lh, angle_deg, center_ys):
                    continue
            patch = self._generate_label_patch(lw, lh, angle_deg)
            pw, ph = patch.size
            img_rgba.paste(patch, (lx - (pw - lw) // 2, ly - (ph - lh) // 2), patch)
        return img_rgba.convert(img.mode), target


# ─────────────────────────────────────────
#  アノテーション描画ヘルパー
# ─────────────────────────────────────────

# category_id ごとの色（足りない場合は循環）
_PALETTE = ["#e05c2a", "#1a6ec4", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]

def draw_annotations(ax, target):
    cat_map = target.get("category_id_to_name", {})
    for i, (box, cat_id) in enumerate(
        zip(target["boxes"].numpy(), target["labels"].numpy())
    ):
        x1, y1, x2, y2 = box
        color = _PALETTE[int(cat_id) % len(_PALETTE)]
        cy = (y1 + y2) / 2.0
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.0, edgecolor=color, facecolor="none",
        ))
        ax.axhline(cy, color=color, linewidth=0.8, linestyle="--", alpha=0.85)
        label_name = cat_map.get(int(cat_id), f"cat{cat_id}")
        ax.text(x2 + 2, cy, f"{label_name}\ncy={int(cy)}px",
                color=color, fontsize=6, va="center")


# ─────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    orig_img = Image.open(IMAGE_PATH).convert("RGB")
    target   = load_coco_target(IMAGE_PATH, COCO_PATH)

    print(f"image : {IMAGE_PATH}  ({orig_img.size[0]}×{orig_img.size[1]})")
    print(f"annotations: {len(target['boxes'])} boxes")
    cat_map = target.get("category_id_to_name", {})
    for box, cat_id in zip(target["boxes"].numpy(), target["labels"].numpy()):
        print(f"  [{cat_map.get(int(cat_id), cat_id)}]  xyxy={box.tolist()}")

    n_cols = 1 + N_SAMPLES
    n_rows = len(CONFIGS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 5.0 * n_rows),
        gridspec_kw={"wspace": 0.04, "hspace": 0.20},
    )
    if n_rows == 1:
        axes = [axes]

    for row, cfg in enumerate(CONFIGS):
        transform = RandomWhiteLabelPaste(**cfg["kwargs"])
        for col in range(n_cols):
            ax = axes[row][col]
            ax.axis("off")
            if col == 0:
                ax.imshow(orig_img)
                draw_annotations(ax, target)
                if row == 0:
                    ax.set_title("元画像", fontsize=8)
                ax.set_ylabel(cfg["label"], fontsize=7, rotation=0,
                              labelpad=60, va="center")
            else:
                aug_img, aug_target = transform(orig_img.copy(), target)
                ax.imshow(aug_img)
                draw_annotations(ax, aug_target)
                if row == 0:
                    ax.set_title(f"sample {col}", fontsize=8)

    # 凡例をカテゴリ名から動的生成
    cat_map = target.get("category_id_to_name", {})
    legend_elements = [
        mpatches.Patch(facecolor="none",
                       edgecolor=_PALETTE[cid % len(_PALETTE)],
                       label=f"{name} bbox / 中心線")
        for cid, name in sorted(cat_map.items())
    ]
    legend_elements.append(
        mpatches.Patch(facecolor="white", edgecolor="#aaa", label="貼付ラベル（白矩形）")
    )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("RandomWhiteLabelPaste — 各設定での拡張結果", fontsize=10, y=1.01)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nsaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()