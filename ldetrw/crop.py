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
    血清上端(category_id=1) or 下端(category_id=2) の「基準線」(bboxを2分割する
    横線)を中心に、y軸方向にランダムオフセットを与えた256x256正方形クロップを
    生成するクラス。

    - x軸方向はデータセット全体で容器位置が一定であることを前提に、
      crop幅=画像幅(オフセットなし)とする。
    - y軸方向は基準線 ± max_offset の範囲でランダムにcrop中心をずらす。
    - crop領域が画像の上端/下端をはみ出す場合は、はみ出した側の端を画像端に
      固定する(paddingなし)。crop_size < image_height であることを前提に、
      上下同時にはみ出すことはないため、単純なclampのみで両条件を満たせる。
    - 対象カテゴリ以外のbbox(もう一方の線)はアノテーションから除外し、
      当該カテゴリのみを検出対象とする単一クラスのstage2データセットを作る。

    Args:
        crop_size (int): 正方形クロップの一辺(デフォルト256)
        max_offset (int): 基準線を中心にy方向へずらす最大オフセット(px)
        target_type (str): 'top' または 'bottom'
    """

    CATEGORY_MAP = {"top": 1, "bottom": 2}
    
    def __init__(self, crop_size=256, max_offset=64, target_type="top"):
        assert target_type in self.CATEGORY_MAP, \
            f"target_type must be 'top' or 'bottom', got {target_type}"
        self.crop_size = crop_size
        self.max_offset = max_offset
        self.target_type = target_type
        self.category_id = self.CATEGORY_MAP[target_type]

    def _filter_target_category(self, target):
        """
        対象カテゴリのbboxのみを残す。
        label_id=0は背景(no object)を表す規約のため、
        前景クラスは1始まり。単一前景クラスとして学習するにあたり、
        - top (category_id=1) はそのままlabel_id=1
        - bottom (category_id=2) はlabel_id=1にリマップ
        する。
        """
        labels = target["labels"]
        keep = (labels == self.category_id)

        filtered = target.copy()
        fields = ["labels", "boxes", "area", "iscrowd"]
        for field in fields:
            if field in filtered:
                filtered[field] = filtered[field][keep]

        if self.target_type == "bottom":
            # category_id=2 -> label_id=1 にリマップ
            filtered["labels"] = torch.ones_like(filtered["labels"])
        # target_type == "top" の場合は category_id=1 がそのまま
        # label_id=1 として使えるためリマップ不要

        return filtered

    def _get_basis_line_y(self, target):
        """対象カテゴリのbboxから基準線(bboxを2分割する横線)のy座標を算出"""
        boxes = target["boxes"]
        if boxes.shape[0] == 0:
            return None
        # 対象カテゴリのbboxが複数存在する場合は最初の1つを使用
        # (通常は1画像につき1つのはず)
        box = boxes[0]
        y1, y2 = box[1].item(), box[3].item()
        basis_y = (y1 + y2) / 2.0
        return basis_y

    def __call__(self, img, target):
        image_width, image_height = img.size

        # 1. 対象カテゴリ(top or bottom)のみに絞り込む
        target = self._filter_target_category(target)

        basis_y = self._get_basis_line_y(target)
        if basis_y is None:
            raise ValueError(
                f"target_type='{self.target_type}' "
                f"(category_id={self.category_id}) のアノテーションが"
                f"この画像に存在しません。"
            )

        # 2. y方向にランダムオフセットを付与してcrop中心を決定
        offset = random.uniform(-self.max_offset, self.max_offset)
        crop_center_y = basis_y + offset
        crop_top = int(round(crop_center_y - self.crop_size / 2.0))

        # 3. 画像端をはみ出す場合はクリップ(paddingなし)
        #    crop_size < image_height の前提のもとでは、上端・下端の
        #    はみ出しは同時に発生しないため、単純なclampで
        #    「一方でシフトしたらもう一方は判定しない」を自動的に満たす
        max_top = image_height - self.crop_size
        crop_top = max(0, min(crop_top, max_top))

        # 4. x方向はcrop幅=画像幅なのでオフセットなし
        crop_left = int(round((image_width - self.crop_size) / 2.0))

        region = (crop_top, crop_left, self.crop_size, self.crop_size)
        return crop(img, target, region)