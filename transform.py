import random
import math
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw


class RandomWhiteLabelPaste(object):
    """
    血清検体容器画像に白い矩形ラベルをランダムに貼り付けるデータ拡張。

    実際の検体容器に巻かれているラベルを模倣し、
    ラベルに遮られた状態でも血清境界を検出できるよう汎化性能を高める。

    Args:
        num_labels (tuple): 貼り付けるラベル数の範囲 (min, max)
        label_width_ratio (tuple): ラベル幅の画像幅に対する比率範囲 (min, max)
        label_height_ratio (tuple): ラベル高さの画像高さに対する比率範囲 (min, max)
        fill_color (tuple): ラベルの塗りつぶし色 (R, G, B)。デフォルトは白
        border_color (tuple or None): ラベル枠線の色。Noneで枠線なし
        border_width (int): ラベル枠線の太さ (px)
        rotation_range (tuple): ラベルの回転角度範囲 (degrees)。容器に巻かれたラベルは
                                 わずかに傾く場合があるため (-5, 5) 程度を推奨
        p (float): この拡張を適用する確率
        min_bbox_visibility (float): bboxの中心水平線（評価軸）がラベルに
                                      遮蔽されることを許容する割合。
                                      0.0=完全遮蔽を許容, 1.0=一切許容しない
    """

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
    ):
        assert isinstance(num_labels, (list, tuple)) and len(num_labels) == 2
        assert isinstance(label_width_ratio, (list, tuple)) and len(label_width_ratio) == 2
        assert isinstance(label_height_ratio, (list, tuple)) and len(label_height_ratio) == 2
        assert 0.0 <= p <= 1.0

        self.num_labels = num_labels
        self.label_width_ratio = label_width_ratio
        self.label_height_ratio = label_height_ratio
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = border_width
        self.rotation_range = rotation_range
        self.p = p
        self.min_bbox_visibility = min_bbox_visibility

    def _get_bbox_center_lines(self, target, img_h):
        """
        各bboxの中心を2分割する横線のy座標を返す。
        評価指標がbbox中心線のピクセル差であるため、
        この線がラベルで隠れないよう制御するオプションに使用する。

        bboxのフォーマット: [x_min, y_min, x_max, y_max] (xyxy)
        """
        center_ys = []
        if target is not None and "boxes" in target:
            for box in target["boxes"]:
                y_min, y_max = box[1].item(), box[3].item()
                center_ys.append((y_min + y_max) / 2.0)
        return center_ys

    def _label_covers_center_line(self, lx, ly, lw, lh, angle_deg, center_ys):
        """
        ラベル矩形（回転あり）がbboxの中心線を覆っているか判定する。
        簡略化のため、回転を無視した軸並行矩形で判定する（小角度では十分な精度）。
        """
        if not center_ys:
            return False
        for cy in center_ys:
            if ly <= cy <= ly + lh:
                return True
        return False

    def _generate_label_patch(self, lw, lh, angle_deg):
        """
        白いラベルパッチ画像（RGBA）を生成する。
        枠線と微細なテキスト模倣ラインを追加して実際のラベルに近づける。
        """
        patch = Image.new("RGBA", (lw, lh), (*self.fill_color, 255))
        draw = ImageDraw.Draw(patch)

        # 枠線
        if self.border_color is not None:
            draw.rectangle(
                [0, 0, lw - 1, lh - 1],
                outline=(*self.border_color, 255),
                width=self.border_width,
            )

        # ラベル内の横線（印字されたテキストを模倣）
        num_lines = random.randint(1, max(1, lh // 14))
        line_color = (180, 180, 180, 120)
        for i in range(num_lines):
            y_line = random.randint(6, max(6, lh - 6))
            line_width_factor = random.uniform(0.3, 0.85)
            x_start = random.randint(4, max(4, int(lw * 0.05)))
            x_end = int(lw * line_width_factor) + x_start
            x_end = min(x_end, lw - 4)
            if x_end > x_start:
                draw.line([(x_start, y_line), (x_end, y_line)], fill=line_color, width=1)

        # 回転
        if angle_deg != 0:
            patch = patch.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

        return patch

    def __call__(self, img, target=None):
        """
        Args:
            img (PIL.Image): 入力画像
            target (dict or None): アノテーション辞書
                - "boxes": torch.Tensor, shape (N, 4), フォーマット [x1, y1, x2, y2]
                - "area": torch.Tensor, shape (N,)
                - "size": torch.Tensor, shape (2,) [H, W]
                - その他のキーはそのまま保持
        Returns:
            (PIL.Image, dict or None): 拡張後の画像とターゲット（bboxは変更なし）
        """
        if random.random() > self.p:
            return img, target

        img_w, img_h = img.size
        img_rgba = img.convert("RGBA")

        # 評価基準となるbbox中心線のy座標を取得
        center_ys = self._get_bbox_center_lines(target, img_h)

        n_labels = random.randint(self.num_labels[0], self.num_labels[1])

        for _ in range(n_labels):
            lw = int(img_w * random.uniform(*self.label_width_ratio))
            lh = int(img_h * random.uniform(*self.label_height_ratio))
            lw = max(lw, 10)
            lh = max(lh, 8)

            angle_deg = random.uniform(*self.rotation_range)

            # ラベルのx位置：容器に巻かれたラベルを模倣するため左端からはみ出しを許容
            lx = random.randint(-lw // 4, max(0, img_w - lw * 3 // 4))
            ly = random.randint(0, max(0, img_h - lh))

            # min_bbox_visibility > 0 の場合、中心線を覆うラベルをスキップ
            if self.min_bbox_visibility > 0.0:
                if self._label_covers_center_line(lx, ly, lw, lh, angle_deg, center_ys):
                    continue

            patch = self._generate_label_patch(lw, lh, angle_deg)

            # 回転で拡張されたpatchサイズを取得し、貼り付け位置を調整
            pw, ph = patch.size
            paste_x = lx - (pw - lw) // 2
            paste_y = ly - (ph - lh) // 2

            img_rgba.paste(patch, (paste_x, paste_y), patch)

        augmented_img = img_rgba.convert(img.mode)

        # targetのbboxは変更しない（ラベルで隠れても正解位置は変わらないため）
        return augmented_img, target