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
GT_CLASS_COLORS = {1: 'yellow', 2: 'cyan'}

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
    from util.box_ops import box_cxcywh_to_xyxy

    out_logits = outputs['pred_logits']  # [B, Q, C=3]
    out_bbox   = outputs['pred_boxes']   # [B, Q, 4]

    prob = out_logits.sigmoid()
    B, Q, C = prob.shape

    b = 0
    img_h, img_w = orig_image_sizes[b]
    scale = torch.stack([img_w, img_h, img_w, img_h]).float()

    result_per_class = {}
    for c in range(C):
        # ==============================================================
        # 修正: index 0は背景（常に低スコア）なのでスキップ
        # 旧: category_id = c + 1（スキップなし）
        # 新: c=0をスキップ、c=1→category_id=1(top)、c=2→category_id=2(bottom)
        # ==============================================================
        if c == 0:
            continue  # index 0 = 背景、スキップ

        category_id = c  # index 1 → id=1(top), index 2 → id=2(bottom)

        topk_k = min(k, Q)
        topk_scores, topk_idx = torch.topk(prob[b, :, c], topk_k)

        boxes = out_bbox[b, topk_idx]
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale[None, :]

        result_per_class[category_id] = {
            'scores':    topk_scores.cpu(),
            'boxes':     boxes.cpu(),
            'query_idx': topk_idx.cpu(),
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
        print(f"  Class {cat_id} ({CLASSES.get(cat_id, str(cat_id))}):")
        for rank in range(min(k, len(info['scores']))):
            score    = info['scores'][rank].item()
            query_id = info['query_idx'][rank].item()  # 追加
            print(f"    top{rank+1}: score={score:.4f}, query_idx={query_id}")
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

                # draw.rectangle([xmin, ymin, xmax, ymax],
                #                 outline="white", width=3)

                gt_color = GT_CLASS_COLORS.get(cat_id, 'white')

                y_center = (ymin + ymax) // 2
                draw.line([xmin, y_center, xmax, y_center], fill="white", width=3)
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

            # draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            
            y_center = (ymin + ymax) // 2
            draw.line([xmin, y_center, xmax, y_center], fill=color, width=3)
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
        H = 48
        W = 8

        # headを並べるグリッドのレイアウトを決定
        n_cols = 4
        n_rows = (num_heads + n_cols - 1) // n_cols  # 切り上げ

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        im = None
        for head_idx in range(num_heads):
            attn_map = attn[head_idx].mean(dim=0).reshape(H, W)
            attn_map = attn_map.cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            ax = axes[head_idx]
            ax.imshow(original_image)
            # imshowの戻り値を保持（colorbar用）
            im = ax.imshow(attn_map,
                        extent=[0, img_w, img_h, 0],
                        alpha=0.5, cmap='jet', interpolation='bilinear')
            ax.set_title(f"head {head_idx}", fontsize=10)
            ax.axis('off')

        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f"Encoder Global Attention: {name}", fontsize=14)
        plt.tight_layout()

        # ==============================================================
        # 修正: 画像右側に1つだけcolorbarを追加
        # 旧: plt.colorbar()なし
        # 新: fig.colorbar()でfig全体の右端に配置
        # ==============================================================
        fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)

        save_path = output_dir / f"attn_enc_{name}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # # 余ったサブプロットを非表示
        # for idx in range(num_heads, len(axes)):
        #     axes[idx].axis('off')

        # fig.suptitle(f"Encoder Global Attention: {name}", fontsize=14)
        # plt.tight_layout()

        # save_path = output_dir / f"attn_enc_{name}.jpg"
        # plt.savefig(save_path, bbox_inches='tight', dpi=150)
        # plt.close()
        print(f"[Encoder Attn] saved: {save_path}")

        # attn_mean = attn.mean(dim=0)
        # attn_map  = attn_mean.mean(dim=0).reshape(H, W)

        # attn_map = attn_map.cpu().numpy()
        # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # ax.imshow(original_image)
        # ax.imshow(attn_map, extent=[0, img_w, img_h, 0],
        #           alpha=0.5, cmap='jet', interpolation='bilinear')
        # ax.set_title(f"Encoder Global Attention: {name}", fontsize=12)
        # ax.axis('off')
        # plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
        #              ax=ax, fraction=0.03, pad=0.04)
        # save_path = output_dir / f"attn_enc_{name}.jpg"
        # plt.savefig(save_path, bbox_inches='tight', dpi=150)
        # plt.close()
        # print(f"[Encoder Attn] saved: {save_path}")

def visualize_decoder_attention(image_path, dec_attn_info_list, output_dir,
                                 result_per_class=None):
    """
    Decoder Attention可視化（クラス別・パターン別）。
 
    出力: レイヤーごとに 2クラス × 4パターン = 8枚
        attn_dec_layer{N}_{cat_id}_all.jpg
        attn_dec_layer{N}_{cat_id}_top1.jpg
        attn_dec_layer{N}_{cat_id}_top2.jpg
        attn_dec_layer{N}_{cat_id}_top3.jpg
 
    各画像レイアウト（横並び2列）:
        左: 論文スタイル（sampling点＋reference点＋bbox）
        右: sampling heatmap（attention weight累積）
 
    描画仕様（論文 Deformable DETR Fig.6 (b) 準拠）:
        - Sampling point : 塗りつぶし円、色=attention weight（青=低 → 赤=高）
        - Reference point: 緑の十字マーカー（marker='P'）
        - BBox           : 緑の矩形（top-kパターンのみ）
        - ラベル         : "カテゴリ スコア" をBBox左上に表示（top-kパターンのみ）
        - Heatmap        : attention weight累積を jet カラーマップで表示
 
    result_per_class の期待フォーマット:
        {
            cat_id: {
                'query_idx': LongTensor [N],   # スコア降順のクエリindex
                'scores'   : FloatTensor [N],  # 信頼スコア（省略可）
                'boxes'    : FloatTensor [N,4], # 正規化(cx,cy,w,h)（省略可）
            }
        }
    """
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size
 
    WEIGHT_CMAP = plt.get_cmap('coolwarm')  # 青(低) → 赤(高)
    HEATMAP_CMAP = plt.get_cmap('jet')
    TOP_QUERIES = 3
 
    # result_per_class が None または空のときは早期リターン
    if not result_per_class:
        print("[Decoder Attn] result_per_class が空のためスキップします。")
        return
 
    # ==============================================================
    # result_per_class を numpy に展開してキャッシュ
    # class_info[cat_id] = {
    #     'query_idx': np.ndarray [N],
    #     'scores'   : np.ndarray [N] or None,
    #     'boxes'    : np.ndarray [N,4] or None,
    # }
    # ==============================================================
    class_info = {}
    for cat_id, res_info in result_per_class.items():
        class_info[cat_id] = {
            'query_idx': res_info['query_idx'][:TOP_QUERIES].numpy(),
            'scores'   : (res_info['scores'][:TOP_QUERIES].numpy()
                          if 'scores' in res_info else None),
            'boxes'    : (res_info['boxes'][:TOP_QUERIES].numpy()
                          if 'boxes' in res_info else None),
        }
 
    # ==============================================================
    # レイヤーループ
    # ==============================================================
    for layer_idx, info in enumerate(dec_attn_info_list):
        ref_points         = info['ref_points'][0]         # [Q, n_levels, 4]
        attn_weights       = info['attn_weights'][0]       # [Q, n_heads, n_levels*n_points]
        sampling_locations = info['sampling_locations'][0] # [Q, n_heads, n_levels, n_points, 2]
 
        Q, n_heads, n_levels, n_points, _ = sampling_locations.shape
 
        # head方向を平均
        attn_w        = attn_weights.reshape(Q, n_heads, n_levels, n_points).mean(dim=1)
        sampling_locs = sampling_locations.mean(dim=1)
        attn_w_np     = attn_w.cpu().numpy()       # [Q, n_levels, n_points]
        sampling_np   = sampling_locs.cpu().numpy() # [Q, n_levels, n_points, 2]
        ref_np        = ref_points.cpu().numpy()    # [Q, n_levels, 4]
 
        # ==============================================================
        # 描画ユーティリティ
        # ==============================================================
 
        def draw_sampling_points(ax, q_indices, alpha=0.7, point_size=60):
            """
            Sampling pointを塗りつぶし円で描画。
            色は描画対象クエリ内で正規化したattention weight。
            全点をまとめて1回の ax.scatter で描画しメモリ・速度を改善。
            """
            qs = np.array(q_indices if q_indices is not None else list(range(Q)))
 
            # [len(qs), n_levels, n_points] のスライスをまとめて取得
            locs_q = sampling_np[qs]   # [len(qs), n_levels, n_points, 2]
            w_q    = attn_w_np[qs]     # [len(qs), n_levels, n_points]
 
            # 全点を1次元に展開
            xs = (locs_q[..., 0] * img_w).ravel()  # [len(qs)*n_levels*n_points]
            ys = (locs_q[..., 1] * img_h).ravel()
            ws = w_q.ravel()
 
            # attention weightを[0,1]に正規化してRGBA配列に変換
            w_min, w_max = ws.min(), ws.max()
            ws_norm = (ws - w_min) / (w_max - w_min + 1e-8)
            colors = WEIGHT_CMAP(ws_norm)  # [N, 4] RGBA
 
            # 1回のscatterで全点を描画
            ax.scatter(xs, ys, s=point_size, c=colors,
                       alpha=alpha, zorder=3, linewidths=0)
 
        def draw_reference_points(ax, q_indices, marker_size=120):
            """Reference pointを緑の十字（marker='P'）で描画。level 0 の (cx,cy) を使用。"""
            qs = q_indices if q_indices is not None else list(range(Q))
            xs = [ref_np[q, 0, 0] * img_w for q in qs]
            ys = [ref_np[q, 0, 1] * img_h for q in qs]
            ax.scatter(xs, ys, s=marker_size, marker='P',
                       color='limegreen', edgecolors='black', linewidths=0.5,
                       zorder=5, label='ref point')
 
        def draw_bboxes(ax, bbox_list):
            """
            BBoxとラベルを描画。
            bbox_list: list of (cx, cy, w, h, label_str)  ← 正規化座標
            """
            for cx, cy, bw, bh, label_str in bbox_list:
                x0 = (cx - bw / 2) * img_w
                y0 = (cy - bh / 2) * img_h
                rect = plt.Rectangle(
                    (x0, y0), bw * img_w, bh * img_h,
                    linewidth=2, edgecolor='limegreen', facecolor='none', zorder=4
                )
                ax.add_patch(rect)
                ax.text(x0, y0 - 4, label_str,
                        fontsize=9, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='limegreen',
                                  ec='none', alpha=0.85),
                        va='bottom', ha='left', zorder=6)
 
        def make_heatmap(q_indices):
            """
            Sampling pointのattention weightを空間に累積したheatmap。
            numpy_index_add 相当の操作でPythonループを排除し高速化。
            各点を中心とした (2r+1)×(2r+1) の矩形領域にweightを加算。
            """
            r = 8
            qs = np.array(q_indices if q_indices is not None else list(range(Q)))
 
            # 全点の座標・重みをまとめて取得
            locs_q = sampling_np[qs]  # [len(qs), n_levels, n_points, 2]
            w_q    = attn_w_np[qs]    # [len(qs), n_levels, n_points]
 
            xs = np.clip((locs_q[..., 0] * img_w).ravel().astype(np.int32), 0, img_w - 1)
            ys = np.clip((locs_q[..., 1] * img_h).ravel().astype(np.int32), 0, img_h - 1)
            ws = w_q.ravel()
 
            # 各点について (2r+1)^2 個のピクセルオフセットを生成してまとめて加算
            dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
            dy = dy.ravel()  # [(2r+1)^2]
            dx = dx.ravel()
 
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
 
            # 各点の矩形領域をブロードキャストで一括計算
            # xs/ys: [N],  dx/dy: [K] → ys_all/xs_all: [N, K]
            ys_all = np.clip(ys[:, None] + dy[None, :], 0, img_h - 1)  # [N, K]
            xs_all = np.clip(xs[:, None] + dx[None, :], 0, img_w - 1)  # [N, K]
 
            # フラットインデックスに変換して np.add.at で累積加算
            flat_idx = (ys_all * img_w + xs_all).ravel()          # [N*K]
            weights  = np.repeat(ws, len(dy))                      # [N*K]
            np.add.at(heatmap.ravel(), flat_idx, weights)
 
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap
 
        def render_and_save(title, save_path, q_indices, bbox_list,
                            point_size, alpha):
            """
            左: 論文スタイル（sampling点 + reference点 + bbox）
            右: sampling heatmap
            の2列レイアウトで保存。
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                                     gridspec_kw={'width_ratios': [1, 1]})
 
            # ---- 左: 論文スタイル ----
            axes[0].imshow(original_image)
            draw_sampling_points(axes[0], q_indices, alpha=alpha,
                                 point_size=point_size)
            draw_reference_points(axes[0], q_indices)
            if bbox_list:
                draw_bboxes(axes[0], bbox_list)
 
            # sampling pointのカラーバー（左パネル右端）
            sm = plt.cm.ScalarMappable(cmap=WEIGHT_CMAP,
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar_l = fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.02)
            cbar_l.set_label('attn weight (norm.)', fontsize=8)
            cbar_l.ax.tick_params(labelsize=7)
 
            axes[0].set_title('Sampling & Reference Points', fontsize=10)
            axes[0].axis('off')
 
            # ---- 右: Heatmap ----
            heatmap = make_heatmap(q_indices)
            axes[1].imshow(original_image)
            im = axes[1].imshow(heatmap, alpha=0.55, cmap=HEATMAP_CMAP,
                                interpolation='bilinear')
            if bbox_list:
                draw_bboxes(axes[1], bbox_list)
 
            cbar_r = fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)
            cbar_r.set_label('attn weight (norm.)', fontsize=8)
            cbar_r.ax.tick_params(labelsize=7)
 
            axes[1].set_title('Sampling Heatmap', fontsize=10)
            axes[1].axis('off')
 
            fig.suptitle(title, fontsize=12, y=1.01)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"[Decoder Attn] saved: {save_path}")
 
        # ==============================================================
        # クラスループ: 各クラスについて 4パターン（all, top1, top2, top3）を出力
        # ==============================================================
        for cat_id, cinfo in class_info.items():
            class_name = CLASSES.get(cat_id, str(cat_id))
            q_idxs     = cinfo['query_idx']   # np.ndarray [TOP_QUERIES]
            scores     = cinfo['scores']       # np.ndarray or None
            boxes      = cinfo['boxes']        # np.ndarray [TOP_QUERIES,4] or None
 
            # ----------------------------------------------------------
            # パターン定義: list of (fname_suffix, title_suffix,
            #                        q_indices, bbox_list,
            #                        point_size, alpha)
            # ----------------------------------------------------------
            patterns = []
 
            # ① all: クラス問わず全クエリを描画（bbox・ラベルなし）
            patterns.append(dict(
                fname_suffix = 'all',
                title_suffix = 'All Queries',
                q_indices    = None,   # 全クエリ
                bbox_list    = [],
                point_size   = 15,
                alpha        = 0.5,
            ))
 
            # ② top-1 / top-2 / top-3: そのクラスの上位クエリを1つずつ描画
            for rank in range(TOP_QUERIES):
                if rank >= len(q_idxs):
                    continue
                q_idx = int(q_idxs[rank])
 
                # ラベル文字列
                score_str = (f"{scores[rank]:.3f}" if scores is not None else "")
                label_str = f"{class_name} {score_str}".strip()
 
                # BBoxリスト
                bbox_list = []
                if boxes is not None:
                    cx, cy, bw, bh = boxes[rank]
                    bbox_list.append((cx, cy, bw, bh, label_str))
 
                patterns.append(dict(
                    fname_suffix = f'top{rank+1}',
                    title_suffix = f"Top-{rank+1} | q={q_idx} | {label_str}",
                    q_indices    = [q_idx],
                    bbox_list    = bbox_list,
                    point_size   = 60,
                    alpha        = 0.8,
                ))
 
            # ----------------------------------------------------------
            # 描画・保存
            # ----------------------------------------------------------
            for pat in patterns:
                title = (f"Dec Layer {layer_idx+1} | "
                         f"{class_name} (cat={cat_id}) | "
                         f"{pat['title_suffix']}")
                save_path = (output_dir /
                             f"attn_dec_layer{layer_idx+1}"
                             f"_cat{cat_id}_{pat['fname_suffix']}.jpg")
                render_and_save(
                    title      = title,
                    save_path  = save_path,
                    q_indices  = pat['q_indices'],
                    bbox_list  = pat['bbox_list'],
                    point_size = pat['point_size'],
                    alpha      = pat['alpha'],
                )
        


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
            result_per_class=result_per_class,
        )
    else:
        print("[INFO] dec_attn_info not in outputs. Skip decoder attention visualization.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR infer script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

#####lW-DETR forward()
def forward(self, samples, targets=None):
    # ...（前半変更なし）...

    # ==============================================================
    # 変更: transformerがdec_attn_infoも返すようになった
    # 旧: hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(...)
    # 新: hs, ref_unsigmoid, hs_enc, ref_enc, dec_attn_info = self.transformer(...)
    # ==============================================================
    hs, ref_unsigmoid, hs_enc, ref_enc, dec_attn_info = self.transformer(
        srcs, masks, poss, refpoint_embed_weight, query_feat_weight
    )

    # ...（bbox/class embed変更なし）...

    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

    if self.aux_loss:
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

    if self.two_stage:
        # ...（変更なし）...

    # ==============================================================
    # 追加: 推論時のみdec_attn_infoをoutputsに含める
    # 学習時はメモリ節約のため含めない
    # ==============================================================
    if not self.training:
        out['dec_attn_info'] = dec_attn_info

    return out

### Transformer forward()
def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
    # ...（前半変更なし）...

    # ==============================================================
    # 変更: decoderがdec_attn_infoも返すようになった
    # 旧: hs, references = self.decoder(...)
    # 新: (hs, references), dec_attn_info = self.decoder(...)
    # ==============================================================
    (hs, references), dec_attn_info = self.decoder(
        tgt, memory,
        memory_key_padding_mask=mask_flatten,
        pos=lvl_pos_embed_flatten,
        refpoints_unsigmoid=refpoint_embed,
        level_start_index=level_start_index,
        spatial_shapes=spatial_shapes,
        valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios
    )

    if self.two_stage:
        if self.bbox_reparam:
            return hs, references, memory_ts, boxes_ts, dec_attn_info
        else:
            return hs, references, memory_ts, boxes_ts.sigmoid(), dec_attn_info

    # ==============================================================
    # 変更: dec_attn_infoを追加して返す
    # 旧: return hs, references, None, None
    # 新: return hs, references, None, None, dec_attn_info
    # ==============================================================
    return hs, references, None, None, dec_attn_info

#### TransformerDecoder forward()
def forward(self, tgt, memory, ...):
    output = tgt
    intermediate = []
    hs_refpoints_unsigmoid = [refpoints_unsigmoid]

    # ==============================================================
    # 追加: 層ごとのdeformable attention情報を格納するリスト
    # 各要素はdict:
    #   'ref_points':        [B, Q, n_levels, 4]  各層のreference point
    #   'attn_weights':      [B, Q, n_heads, n_levels*n_points]
    #   'sampling_locations':[B, Q, n_heads, n_levels, n_points, 2]
    # ==============================================================
    dec_attn_info = []

    if self.lite_refpoint_refine:
        if self.bbox_reparam:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
        else:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

    for layer_id, layer in enumerate(self.layers):
        if not self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        pos_transformation = 1
        query_pos = query_pos * pos_transformation

        # ==============================================================
        # 変更: layerの戻り値にattn情報を追加
        # 旧: output = layer(output, memory, ...)
        # 新: output, ref_pts, attn_w, sampling_locs = layer(output, memory, ...)
        # ==============================================================
        output, ref_pts, attn_w, sampling_locs = layer(
            output, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=(layer_id == 0),
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        # ==============================================================
        # 追加: 層ごとのattn情報を保存
        # ==============================================================
        dec_attn_info.append({
            'ref_points':         ref_pts.detach(),        # [B, Q, n_levels, 4]
            'attn_weights':       attn_w.detach(),         # [B, Q, n_heads, n_levels*n_points]
            'sampling_locations': sampling_locs.detach(),  # [B, Q, n_heads, n_levels, n_points, 2]
        })

        if not self.lite_refpoint_refine:
            new_refpoints_delta = self.bbox_embed(output)
            new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
            if layer_id != self.num_layers - 1:
                hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
            refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

        if self.return_intermediate:
            intermediate.append(self.norm(output))

    if self.norm is not None:
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

    if self.return_intermediate:
        if self.bbox_embed is not None:
            # ==============================================================
            # 変更: dec_attn_infoを追加して返す
            # 旧: return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)]
            # 新: return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)], dec_attn_info
            # ==============================================================
            return [
                torch.stack(intermediate),
                torch.stack(hs_refpoints_unsigmoid),
            ], dec_attn_info
        else:
            return [
                torch.stack(intermediate),
                refpoints_unsigmoid.unsqueeze(0)
            ], dec_attn_info

    return output.unsqueeze(0), dec_attn_info
 
 #### TransformerDecoderLayer.forward_post()
 def forward_post(self, tgt, memory,
                 tgt_mask=None, memory_mask=None,
                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                 pos=None, query_pos=None, query_sine_embed=None,
                 is_first=False, reference_points=None,
                 spatial_shapes=None, level_start_index=None):
    bs, num_queries, _ = tgt.shape

    # Self-Attention（変更なし）
    q = k = tgt + query_pos
    v = tgt
    if self.training:
        q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
        k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
        v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)
    tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    if self.training:
        tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    # ==============================================================
    # 変更: cross_attnがattention_weights, sampling_locationsも
    # 返すように変更（MSDeformAttn側の改変が別途必要）
    # 旧: tgt2 = self.cross_attn(...)
    # 新: tgt2, attn_weights, sampling_locations = self.cross_attn(...)
    # ==============================================================
    tgt2, attn_weights, sampling_locations = self.cross_attn(
        self.with_pos_embed(tgt, query_pos),
        reference_points,
        memory,
        spatial_shapes,
        level_start_index,
        memory_key_padding_mask
    )

    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)

    # ==============================================================
    # 変更: reference_points, attn_weights, sampling_locationsも返す
    # 旧: return tgt
    # 新: return tgt, reference_points, attn_weights, sampling_locations
    # ==============================================================
    return tgt, reference_points, attn_weights, sampling_locations


def forward(self, tgt, memory, ...):
    # ==============================================================
    # 変更: forward_postの戻り値を全てパススルー
    # 旧: return self.forward_post(...)
    # 新: return self.forward_post(...)  ← タプルをそのまま返す
    # ==============================================================
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask,
                             pos, query_pos, query_sine_embed, is_first,
                             reference_points, spatial_shapes, level_start_index)