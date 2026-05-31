"""
COCO Evaluator for Object Detection
=====================================
- prediction file format: <class_id> <xmin> <ymin> <w> <h> <confidence>
- GT file format:   <class_id> <xmin> <ymin> <w> <h>

usage:
    python coco_evaluator.py \
        --pred_dir  ./predictions \
        --gt_dir    ./ground_truths \
        --img_width 640 \
        --img_height 480 \
        [--num_classes 3] \
        [--iou_threshold 0.5]
"""

import os
import glob
import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_gt_file(filepath: str, img_width: int, img_height: int):
    boxes = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xmin = float(parts[1]) * img_width
            ymin = float(parts[2]) * img_height
            w    = float(parts[3]) * img_width
            h    = float(parts[4]) * img_height
            boxes.append({"class_id": cls, "xmin": xmin, "ymin": ymin, "w": w, "h": h})
    return boxes


def load_pred_file(filepath: str):
    boxes = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls   = int(parts[0])
            xmin  = float(parts[1])
            ymin  = float(parts[2])
            w     = float(parts[3])
            h     = float(parts[4])
            score = float(parts[5])
            boxes.append({"class_id": cls, "xmin": xmin, "ymin": ymin,
                          "w": w, "h": h, "score": score})
    return boxes

def build_coco_gt(gt_dir: str, img_width: int, img_height: int, num_classes: int):
    categories = [{"id": i, "name": str(i)} for i in range(num_classes)]
    images, annotations = [], []
    ann_id = 1
    img_id_map = {}   # stem -> image_id

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        raise FileNotFoundError(f"GT files not found in: {gt_dir}")

    for img_id, filepath in enumerate(gt_files, start=1):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        img_id_map[stem] = img_id
        images.append({"id": img_id, "width": img_width, "height": img_height,
                        "file_name": stem})

        for box in load_gt_file(filepath, img_width, img_height):
            area = box["w"] * box["h"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box["class_id"],
                "bbox": [box["xmin"], box["ymin"], box["w"], box["h"]],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_gt_dict = {"images": images, "annotations": annotations,
                    "categories": categories}
    return coco_gt_dict, img_id_map


def build_coco_predictions(pred_dir: str, img_id_map: dict):
    results = []
    for stem, img_id in img_id_map.items():
        filepath = os.path.join(pred_dir, stem + ".txt")
        if not os.path.exists(filepath):
            continue
        for box in load_pred_file(filepath):
            results.append({
                "image_id": img_id,
                "category_id": box["class_id"],
                "bbox": [box["xmin"], box["ymin"], box["w"], box["h"]],
                "score": box["score"],
            })
    return results

def run_evaluation(gt_dir, pred_dir, img_width, img_height,
                   num_classes=3, iou_threshold=None):

    print("=" * 60)
    print("COCO Evaluator  |  Precision / Recall / mAP")
    print("=" * 60)
    print(f"  GT dir   : {gt_dir}")
    print(f"  Pred dir : {pred_dir}")
    print(f"  Image    : {img_width} x {img_height}")
    print(f"  Classes  : {num_classes}  (ids 0 ~ {num_classes-1})")
    print("=" * 60)

    coco_gt_dict, img_id_map = build_coco_gt(gt_dir, img_width, img_height, num_classes)
    pred_list = build_coco_predictions(pred_dir, img_id_map)

    total_gt  = len(coco_gt_dict["annotations"])
    total_pred = len(pred_list)
    print(f"  GT boxes   : {total_gt}")
    print(f"  Pred boxes : {total_pred}")
    print("=" * 60)

    if total_pred == 0:
        print("Warning")
        return

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_gt_dict, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_list, f)
        pred_path = f.name

    coco_gt   = COCO(gt_path)
    coco_pred = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    if iou_threshold is not None:
        coco_eval.params.iouThrs = np.array([iou_threshold])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n─── Per-Class AP (IoU=0.50) ─────────────────────────")
    cat_ids = coco_gt.getCatIds()
    for cat_id in cat_ids:
        coco_eval_cls = COCOeval(coco_gt, coco_pred, iouType="bbox")
        coco_eval_cls.params.catIds  = [cat_id]
        coco_eval_cls.params.iouThrs = np.array([0.50])
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        ap = coco_eval_cls.eval["precision"]
        # precision shape: [T, R, K, A, M]
        # T=iou, R=recall, K=category, A=area, M=maxDets
        ap_val = ap[0, :, 0, 0, -1]  # IoU=0.5, all recalls, first (only) cat, all area, maxDets
        mean_ap = float(np.mean(ap_val[ap_val > -1])) if np.any(ap_val > -1) else 0.0
        cat_name = coco_gt.loadCats([cat_id])[0]["name"]
        print(f"  Class {cat_id} ({cat_name}): AP@0.50 = {mean_ap:.4f}")

    print("\n─── Overall Precision & Recall @ IoU=0.50 ───────────")
    prec = coco_eval.eval["precision"]   # [T, R, K, A, M]
    rec  = coco_eval.eval["recall"]      # [T, K, A, M]

    iou_idx = 0 if iou_threshold is None else 0

    prec_50 = prec[iou_idx, :, :, 0, -1]   # all recalls, all cats, area=all
    rec_50  = rec[iou_idx, :, 0, -1]        # all cats, area=all

    valid_prec = prec_50[prec_50 > -1]
    mean_prec  = float(np.mean(valid_prec)) if len(valid_prec) > 0 else 0.0
    valid_rec  = rec_50[rec_50 > -1]
    mean_rec   = float(np.mean(valid_rec))  if len(valid_rec) > 0 else 0.0

    print(f"  Mean Precision @ IoU=0.50 : {mean_prec:.4f}")
    print(f"  Mean Recall    @ IoU=0.50 : {mean_rec:.4f}")

    if mean_prec + mean_rec > 0:
        f1 = 2 * mean_prec * mean_rec / (mean_prec + mean_rec)
        print(f"  F1 Score       @ IoU=0.50 : {f1:.4f}")

    print("=" * 60)

    os.unlink(gt_path)
    os.unlink(pred_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO Evaluator: predicted bbox .txt vs GT .txt (normalized)"
    )
    parser.add_argument("--pred_dir",    required=True,  help="予測 .txt フォルダのパス")
    parser.add_argument("--gt_dir",      required=True,  help="GT .txt フォルダのパス  (正規化済み)")
    parser.add_argument("--img_width",   required=True,  type=int, help="画像の幅 (px)")
    parser.add_argument("--img_height",  required=True,  type=int, help="画像の高さ (px)")
    parser.add_argument("--num_classes", default=3,      type=int, help="クラス数 (デフォルト: 3)")
    parser.add_argument("--iou_threshold", default=None, type=float,
                        help="単一 IoU 閾値 (例: 0.5)。省略すると COCO 標準 0.5:0.95 を使用")
    args = parser.parse_args()

    run_evaluation(
        gt_dir        = args.gt_dir,
        pred_dir      = args.pred_dir,
        img_width     = args.img_width,
        img_height    = args.img_height,
        num_classes   = args.num_classes,
        iou_threshold = args.iou_threshold,
    )