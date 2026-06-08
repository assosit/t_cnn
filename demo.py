import argparse
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import util.misc as utils
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.misc import get_param_dict

# ==============================================================
# (1) 修正: backgroundなし、上端・下端の2クラスのみ
# 旧: CLASSES = ['top', 'bottom']  # list、0始まり
# 新: CLASSES = {1: 'top', 2: 'bottom'}  # dict、1始まり（GT category_idに合わせる）
# ==============================================================
CLASSES = {1: 'top', 2: 'bottom'}
# ==============================================================
# (1) 修正: CLASS_COLORSもdictに変更
# 旧: CLASS_COLORS = ['red', 'blue']
# 新: CLASS_COLORS = {1: 'red', 2: 'blue'}
# ==============================================================
CLASS_COLORS = {1: 'red', 2: 'blue'}

# top-Kの数
TOP_K = 5  # (2) 各クラスtop-5を取得


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--weights', type=str, default=None, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrained_encoder', type=str, default=None)
    parser.add_argument('--encoder', default='vit_tiny', type=str)
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int)
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+')
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--sa_nheads', default=8, type=int)
    parser.add_argument('--ca_nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--group_detr', default=13, type=int)
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true')
    parser.add_argument('--num_select', default=100, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_path', type=float, default=0)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--sum_group_losses', action='store_true')
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')
    parser.add_argument('--input', default=None, required=True)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    # ==============================================================
    # (2) 追加: GT読み込み用データセットフォルダパス
    # 旧: なし
    # 新: --dataset_dir でCOCO形式データセットのルートを指定
    # ==============================================================
    parser.add_argument('--dataset_dir', default=None,
                        help='Path to dataset root directory for loading GT annotations.')
    return parser


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize([640, 640]),
        normalize,
    ])
    image = transform(image)
    return image, orig_image_size


# ==============================================================
# (2) 追加: COCOアノテーションから対象画像のGT情報を読み込む関数
# ==============================================================
def load_gt_for_image(dataset_dir, image_path):
    """
    指定した画像ファイルに対応するGT bbox情報を
    COCOアノテーションファイルから読み込む。

    Args:
        dataset_dir: データセットのルートフォルダパス
                     （annotations/*.jsonが存在するフォルダ）
        image_path:  可視化対象の画像ファイルパス

    Returns:
        gt_info: list of dict
            [{'category_id': int, 'bbox': [x_min, y_min, w, h]}, ...]
            bbox はCOCO形式（x_min, y_min, width, height）絶対座標
    """
    dataset_dir = Path(dataset_dir)
    image_filename = Path(image_path).name

    # annotationsフォルダ内の全jsonを検索
    ann_candidates = list(dataset_dir.glob('annotations/*.json'))
    if not ann_candidates:
        raise FileNotFoundError(f"annotations/*.json が見つかりません: {dataset_dir}")

    # 対象画像が含まれるアノテーションファイルを特定
    ann_data = None
    target_image_info = None
    for ann_path in ann_candidates:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        matched = [img for img in data['images']
                   if Path(img['file_name']).name == image_filename]
        if matched:
            ann_data = data
            target_image_info = matched[0]
            break

    if ann_data is None:
        raise ValueError(f"{image_filename} はいずれのアノテーションファイルにも見つかりません。")

    image_id = target_image_info['id']

    # 該当image_idのannotationを抽出
    gt_info = [
        {
            'category_id': ann['category_id'],  # 1始まり
            'bbox': ann['bbox'],                 # [x_min, y_min, w, h]
        }
        for ann in ann_data['annotations']
        if ann['image_id'] == image_id
    ]

    return gt_info


# ==============================================================
# (1)(2) 修正: クラス別top-Kのbboxを取得する関数
# 旧: result_per_class のキーが c（0始まり）
# 新: result_per_class のキーが c+1（1始まり、GT category_idに対応）
# ==============================================================
def get_topk_per_class(outputs, orig_image_sizes, k=TOP_K):
    """
    各クラスについてスコアtop-Kのbbox・スコアを返す。

    Returns:
        result_per_class: dict
            {
              category_id(int, 1始まり): {
                'scores':    Tensor [K],   降順ソート済み
                'boxes':     Tensor [K,4], x1y1x2y2・絶対座標
                'query_idx': Tensor [K],   対応するクエリインデックス
              }
            }
    """
    from util.box_ops import box_cxcywh_to_xyxy

    out_logits = outputs['pred_logits']  # [B, Q, C]
    out_bbox   = outputs['pred_boxes']   # [B, Q, 4]

    prob = out_logits.sigmoid()          # [B, Q, C]
    B, Q, C = prob.shape

    # バッチサイズ=1前提（デモ用）
    b = 0
    img_h, img_w = orig_image_sizes[b]
    scale = torch.stack([img_w, img_h, img_w, img_h]).float()

    result_per_class = {}
    for c in range(C):
        topk_k = min(k, Q)
        topk_scores, topk_idx = torch.topk(prob[b, :, c], topk_k)

        boxes = out_bbox[b, topk_idx]
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale[None, :]

        # ==============================================================
        # (1) 修正: キーをc（0始まり）からc+1（1始まり）に変更
        # 旧: result_per_class[c] = ...
        # 新: result_per_class[c + 1] = ...
        # ==============================================================
        category_id = c + 1
        result_per_class[category_id] = {
            'scores':    topk_scores.cpu(),
            'boxes':     boxes.cpu(),
            'query_idx': topk_idx.cpu(),  # sampling point可視化用
        }

    return result_per_class


# ==============================================================
# (1)(2) 修正: top-Kスコアのprint
# 旧: CLASSES[c]（listインデックス）
# 新: CLASSES.get(cat_id)（dictキー参照、1始まり）
# ==============================================================
def print_topk_scores(result_per_class, k=TOP_K):
    print("\n===== Top-K Scores per Class =====")
    for cat_id, info in result_per_class.items():
        # (1) 修正: CLASSES[c] → CLASSES.get(cat_id)
        print(f"  Class {cat_id} ({CLASSES.get(cat_id, str(cat_id))}):")
        for rank in range(min(k, len(info['scores']))):
            score = info['scores'][rank].item()
            print(f"    top{rank+1}: score={score:.4f}")
    print("==================================\n")


# ==============================================================
# (1)(2) 修正: top-1〜top-Kまで順位ごとに1枚ずつ描画して保存
# 旧: CLASSES[c], CLASS_COLORS[c]（listインデックス）
#     gt_info引数なし
# 新: CLASSES.get(cat_id), CLASS_COLORS.get(cat_id)（dictキー参照）
#     gt_info引数を追加しGTも描画
# ==============================================================
def visualize_topk_detections(image_path, result_per_class, output_dir,
                               gt_info=None, k=TOP_K):
    """
    top-1〜top-Kまで順位ごとに画像を1枚ずつ保存する。
    各画像にはクラス1とクラス2のその順位のbboxを描画。
    gt_infoが与えられた場合はGT bboxも描画。

    出力ファイル名: visualize_top{rank}.jpg (rank=1〜K)

    gt_info: load_gt_for_image()の戻り値
        [{'category_id': int, 'bbox': [x_min, y_min, w, h]}, ...]
    """
    output_dir = Path(output_dir)

    for rank in range(k):
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        font = ImageFont.load_default()

        # ----------------------------------------------------------
        # (2) 追加: GT bboxを描画（白枠、予測bboxと区別）
        # ----------------------------------------------------------
        if gt_info is not None:
            for gt in gt_info:
                cat_id = gt['category_id']
                x_min, y_min, w, h = gt['bbox']
                xmin = int(x_min)
                ymin = int(y_min)
                xmax = int(x_min + w)
                ymax = int(y_min + h)

                draw.rectangle([xmin, ymin, xmax, ymax],
                                outline="white", width=3)
                gt_label = f"GT:{CLASSES.get(cat_id, str(cat_id))}"
                draw.text((xmin, max(ymin - 12, 0)),
                           gt_label, fill="white", font=font)

        # ----------------------------------------------------------
        # (1) 修正: 予測bboxの描画
        # 旧: color = CLASS_COLORS[c], label_text = f"{CLASSES[c]} ..."
        # 新: color = CLASS_COLORS.get(cat_id), label_text = f"{CLASSES.get(cat_id)} ..."
        # ----------------------------------------------------------
        for cat_id, info in result_per_class.items():
            if rank >= len(info['scores']):
                continue

            score = info['scores'][rank].item()
            box   = info['boxes'][rank]
            xmin, ymin, xmax, ymax = map(int, box)
            # (1) 修正: dictキー参照に変更
            color      = CLASS_COLORS.get(cat_id, 'green')
            label_text = f"{CLASSES.get(cat_id, str(cat_id))} top{rank+1} {score:.3f}"

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin, max(ymin - 12, 0)), label_text, fill=color, font=font)

        save_path = output_dir / f"visualize_top{rank+1}.jpg"
        original_image.save(save_path)
        print(f"[top{rank+1}] saved: {save_path}")


# ==============================================================
# (3) Encoder global attention mapの可視化（変更なし）
# ==============================================================
def visualize_encoder_attention(image_path, enc_attn_weights_list, output_dir,
                                 layer_names=None):
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size

    if layer_names is None:
        layer_names = [f"enc_global_layer{i+1}" for i in range(len(enc_attn_weights_list))]

    for layer_idx, (attn, name) in enumerate(zip(enc_attn_weights_list, layer_names)):
        attn = attn[0]
        num_heads, HW, _ = attn.shape
        H = W = int(HW ** 0.5)

        attn_mean = attn.mean(dim=0)
        attn_map  = attn_mean.mean(dim=0).reshape(H, W)

        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(original_image)
        ax.imshow(attn_map, extent=[0, img_w, img_h, 0],
                  alpha=0.5, cmap='jet', interpolation='bilinear')
        ax.set_title(f"Encoder Global Attention: {name}", fontsize=12)
        ax.axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                     ax=ax, fraction=0.03, pad=0.04)
        save_path = output_dir / f"attn_enc_{name}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Encoder Attn] saved: {save_path}")


# ==============================================================
# (3) 修正: Decoder deformable cross attention mapの可視化
# 旧: ref_points/attn_weightsのshapeがdec_attn_infoと不一致だった
# 新: sampling_locationsをattn_weightsで重み付けしてheatmap描画
# ==============================================================
def visualize_decoder_attention(image_path, dec_attn_info_list, output_dir):
    """
    dec_attn_info_list: list of dict（decoder層ごと）
      各dict:
        'ref_points':         Tensor [B, Q, n_levels, 4]   正規化cx,cy,w,h
        'attn_weights':       Tensor [B, Q, n_heads, n_levels*n_points]
        'sampling_locations': Tensor [B, Q, n_heads, n_levels, n_points, 2]  正規化xy
    """
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size

    for layer_idx, info in enumerate(dec_attn_info_list):
        # B=1前提でバッチ次元を除去
        ref_points         = info['ref_points'][0]         # [Q, n_levels, 4]
        attn_weights       = info['attn_weights'][0]       # [Q, n_heads, n_levels*n_points]
        sampling_locations = info['sampling_locations'][0] # [Q, n_heads, n_levels, n_points, 2]

        Q, n_heads, n_levels, n_points, _ = sampling_locations.shape

        # head平均
        attn_w       = attn_weights.reshape(Q, n_heads, n_levels, n_points).mean(dim=1)  # [Q, n_levels, n_points]
        sampling_locs = sampling_locations.mean(dim=1)                                    # [Q, n_levels, n_points, 2]

        attn_w_np    = attn_w.cpu().numpy()
        sampling_np  = sampling_locs.cpu().numpy()

        # sampling pointをattn_weightで重み付けしたheatmap
        heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        r = 8
        for q in range(Q):
            for lvl in range(n_levels):
                for p in range(n_points):
                    x = int(np.clip(sampling_np[q, lvl, p, 0] * img_w, 0, img_w - 1))
                    y = int(np.clip(sampling_np[q, lvl, p, 1] * img_h, 0, img_h - 1))
                    w = attn_w_np[q, lvl, p]
                    y0, y1 = max(0, y - r), min(img_h, y + r + 1)
                    x0, x1 = max(0, x - r), min(img_w, x + r + 1)
                    heatmap[y0:y1, x0:x1] += w

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左: attn weight重み付きsampling pointのheatmap
        axes[0].imshow(original_image)
        axes[0].imshow(heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
        axes[0].set_title(f"Dec Layer {layer_idx+1}: Sampling Points (weighted by attn)")
        axes[0].axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                     ax=axes[0], fraction=0.03, pad=0.04)

        # 右: reference pointの散布図（level別色分け）
        axes[1].imshow(original_image)
        colors_level = ['red', 'blue', 'green', 'orange']
        for lvl in range(n_levels):
            cx = ref_points[:, lvl, 0].cpu().numpy() * img_w
            cy = ref_points[:, lvl, 1].cpu().numpy() * img_h
            axes[1].scatter(cx, cy, s=5, alpha=0.4,
                            color=colors_level[lvl % len(colors_level)],
                            label=f"level {lvl}")
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].set_title(f"Dec Layer {layer_idx+1}: Reference Points (per level)")
        axes[1].axis('off')

        save_path = output_dir / f"attn_dec_layer{layer_idx+1}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Decoder Attn] saved: {save_path}")


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    param_dicts = get_param_dict(args, model)

    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    # preprocess
    image, orig_image_size = preprocess_image(args.input)
    image = image.to(device)
    orig_image_size = orig_image_size.to(device)

    images = nested_tensor_from_tensor_list([image])
    orig_image_sizes = torch.stack([orig_image_size])

    with torch.no_grad():
        outputs = model(images)

    # (1)(2) クラス別top-K取得（キーが1始まりになった）
    result_per_class = get_topk_per_class(outputs, orig_image_sizes, k=TOP_K)

    # top-Kスコアをprint
    print_topk_scores(result_per_class, k=TOP_K)

    # ==============================================================
    # (2) 追加: GT読み込み
    # --dataset_dirが指定されている場合のみ実行
    # ==============================================================
    gt_info = None
    if args.dataset_dir is not None:
        try:
            gt_info = load_gt_for_image(args.dataset_dir, args.input)
            print("\n===== GT Info =====")
            for gt in gt_info:
                cat_name = CLASSES.get(gt['category_id'], str(gt['category_id']))
                print(f"  category: {cat_name} (id={gt['category_id']}), "
                      f"bbox={gt['bbox']}")
            print("===================\n")
        except Exception as e:
            print(f"[WARN] GT読み込み失敗: {e}")

    # ==============================================================
    # (2) 修正: gt_infoを渡して描画
    # 旧: visualize_topk_detections(args.input, result_per_class, args.output_dir, k=TOP_K)
    # 新: gt_info引数を追加
    # ==============================================================
    visualize_topk_detections(
        args.input, result_per_class, args.output_dir,
        gt_info=gt_info, k=TOP_K
    )

    if 'enc_attn_weights' in outputs:
        visualize_encoder_attention(
            image_path=args.input,
            enc_attn_weights_list=outputs['enc_attn_weights'],
            output_dir=args.output_dir,
            layer_names=['enc_global_layer1', 'enc_global_layer3', 'enc_global_layer5'],
        )
    else:
        print("[INFO] enc_attn_weights not in outputs. Skip encoder attention visualization.")

    # ==============================================================
    # (3) 修正: dec_attn_infoをoutputsから直接取得
    # 旧: dec_attn_info_list, orig_image_size引数が必要だった
    # 新: orig_image_size不要（sampling_locationsが正規化座標のため）
    # ==============================================================
    if 'dec_attn_info' in outputs:
        visualize_decoder_attention(
            image_path=args.input,
            dec_attn_info_list=outputs['dec_attn_info'],
            output_dir=args.output_dir,
        )
    else:
        print("[INFO] dec_attn_info not in outputs. Skip decoder attention visualization.")

# --------------------------------------------------------------
        # [0,1] Sampling point scatter（上位クエリのみ）
        # 各クエリをtop順に色分け・ラベル付きでプロット
        # --------------------------------------------------------------
        axes[0, 1].imshow(original_image)
        if topk_query_info:
            for cat_id, qi in topk_query_info.items():
                for rank, q_idx in enumerate(qi['query_idx']):
                    # 全level・全pointのsampling locationを散布
                    for lvl in range(n_levels):
                        for p in range(n_points):
                            x = sampling_np[q_idx, lvl, p, 0] * img_w
                            y = sampling_np[q_idx, lvl, p, 1] * img_h
                            # top1のみラベル付き、以降はラベルなし（凡例重複防止）
                            label = f"{qi['label']} top{rank+1} (q={q_idx})" \
                                    if (lvl == 0 and p == 0) else None
                            axes[0, 1].scatter(x, y, s=20,
                                               color=qi['color'],
                                               alpha=0.8 - rank * 0.1,  # 順位が下がるほど薄く
                                               label=label,
                                               marker=f"${rank+1}$")    # 順位番号をマーカーに
        axes[0, 1].legend(loc='upper right', fontsize=7, markerscale=1.5)
        axes[0, 1].set_title(f"Dec Layer {layer_idx+1}: Sampling Points\n(Top-{TOP_K} Queries per Class)")
        axes[0, 1].axis('off')

        # --------------------------------------------------------------
        # [1,0] Reference point scatter（全クエリ、level別色分け）
        # ref_points[q, lvl, :2] = cx, cy（正規化座標）
        # --------------------------------------------------------------
        axes[1, 0].imshow(original_image)
        colors_level = ['red', 'blue', 'green', 'orange']
        for lvl in range(n_levels):
            cx = ref_np[:, lvl, 0] * img_w   # 全クエリのcx
            cy = ref_np[:, lvl, 1] * img_h   # 全クエリのcy
            axes[1, 0].scatter(cx, cy, s=5, alpha=0.4,
                               color=colors_level[lvl % len(colors_level)],
                               label=f"level {lvl}")
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].set_title(f"Dec Layer {layer_idx+1}: Reference Points\n(All Queries, per level)")
        axes[1, 0].axis('off')

        # --------------------------------------------------------------
        # [1,1] Reference point scatter（上位クエリのみ）
        # --------------------------------------------------------------
        axes[1, 1].imshow(original_image)
        if topk_query_info:
            for cat_id, qi in topk_query_info.items():
                for rank, q_idx in enumerate(qi['query_idx']):
                    for lvl in range(n_levels):
                        x = ref_np[q_idx, lvl, 0] * img_w
                        y = ref_np[q_idx, lvl, 1] * img_h
                        label = f"{qi['label']} top{rank+1} (q={q_idx})" \
                                if lvl == 0 else None
                        axes[1, 1].scatter(x, y, s=40,
                                           color=qi['color'],
                                           alpha=0.8 - rank * 0.1,
                                           label=label,
                                           marker=f"${rank+1}$")
        axes[1, 1].legend(loc='upper right', fontsize=7, markerscale=1.5)
        axes[1, 1].set_title(f"Dec Layer {layer_idx+1}: Reference Points\n(Top-{TOP_K} Queries per Class)")
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = output_dir / f"attn_dec_layer{layer_idx+1}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Decoder Attn] saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR infer script', parents=[get_args_parser()])\
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)