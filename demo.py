import argparse
import copy
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
# 旧: COCO_CLASSES = ['__background__', 'person', ...]
# ==============================================================
CLASSES = ['top', 'bottom']  # index 0=上端, 1=下端
CLASS_COLORS = ['red', 'blue']  # 描画色（クラスごとに区別）

# top-Kの数
TOP_K = 5  # (2) 各クラスtop-5を取得


def get_args_parser():
    # --- 変更なし（省略せずそのまま残す） ---
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
    return parser


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # orig_image_size = torch.tensor(image.size[::-1])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize([640, 640]),
        normalize,
    ])
    image = transform(image)
    # 64の倍数でパディングする処理後の画像サイズを求めてorig_image_sizeとする
    # imageはパディングしなくて良い(モデル入力後にパディングするため)


    return image, orig_image_size


# ==============================================================
# (2) 修正: クラス別top-Kのbboxを取得する関数
# 旧: postprocessors['bbox']がflatten top-100を返していた
# 新: クラスごとに独立してtop-K件を取得
# ==============================================================
def get_topk_per_class(outputs, orig_image_sizes, k=TOP_K):
    """
    各クラスについてスコアtop-Kのbbox・スコアを返す。

    Returns:
        result_per_class: dict
            {
              class_id(int): {
                'scores': [k],   降順ソート済み
                'boxes':  [k,4], x1y1x2y2・絶対座標
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
        # クラスcのスコアで降順top-K
        topk_scores, topk_idx = torch.topk(prob[b, :, c], topk_k)

        boxes = out_bbox[b, topk_idx]              # [K, 4]  cx,cy,w,h
        boxes = box_cxcywh_to_xyxy(boxes)          # [K, 4]  x1,y1,x2,y2
        boxes = boxes * scale[None, :]             # 絶対座標に変換

        result_per_class[c] = {
            'scores': topk_scores.cpu(),
            'boxes':  boxes.cpu(),
        }

    return result_per_class


# ==============================================================
# (2) 修正: top-1〜top-Kまで順位ごとに1枚ずつ描画して保存
# 旧: visualize_detections()が1枚の画像にすべて描画していた
# 新: 各順位ごとに1枚（クラス0とクラス1を同じ画像に描画）、合計K枚を保存
# ==============================================================
def visualize_topk_detections(image_path, result_per_class, output_dir, k=TOP_K):
    """
    top-1〜top-Kまで順位ごとに画像を1枚ずつ保存する。
    各画像にはクラス0とクラス1のその順位のbboxを描画。

    出力ファイル名: visualize_top{rank}.jpg (rank=1〜K)
    """
    output_dir = Path(output_dir)

    for rank in range(k):  # rank=0がtop-1
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        font = ImageFont.load_default()

        for c, info in result_per_class.items():
            if rank >= len(info['scores']):
                continue  # クエリ数がK未満の場合のガード

            score = info['scores'][rank].item()
            box   = info['boxes'][rank]
            xmin, ymin, xmax, ymax = map(int, box)
            color = CLASS_COLORS[c]
            label_text = f"{CLASSES[c]} top{rank+1} {score:.3f}"

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin, max(ymin - 12, 0)), label_text, fill=color, font=font)

        save_path = output_dir / f"visualize_top{rank+1}.jpg"
        original_image.save(save_path)
        print(f"[top{rank+1}] saved: {save_path}")


# ==============================================================
# (2) 追加: top-Kスコアをクラスごとにprint
# ==============================================================
def print_topk_scores(result_per_class, k=TOP_K):
    print("\n===== Top-K Scores per Class =====")
    for c, info in result_per_class.items():
        print(f"  Class {c} ({CLASSES[c]}):")
        for rank in range(min(k, len(info['scores']))):
            score = info['scores'][rank].item()
            print(f"    top{rank+1}: score={score:.4f}")
    print("==================================\n")


# ==============================================================
# (3) 追加: Encoder global attention mapの可視化
# ViTのglobal attention層（1,3,5層目=index 0,2,4）の
# attention weightを画像上にheatmapとして重ねて描画
#
# 前提: modelのforward()が以下を返すように改変済みであること
#   outputs['enc_attn_weights']: list of [B, num_heads, HW, HW]
#     index 0,1,2 がそれぞれ global attention 層1,3,5層目
# ==============================================================
def visualize_encoder_attention(image_path, enc_attn_weights_list, output_dir,
                                 layer_names=None):
    """
    enc_attn_weights_list: list of Tensor [B, num_heads, HW, HW]
      各要素がglobal attention層1層分のattention weight
    layer_names: list of str, 各層の名前（例: ['enc_layer1', 'enc_layer3', 'enc_layer5']）
    """
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size

    if layer_names is None:
        layer_names = [f"enc_global_layer{i+1}" for i in range(len(enc_attn_weights_list))]

    for layer_idx, (attn, name) in enumerate(zip(enc_attn_weights_list, layer_names)):
        # attn: [B, num_heads, HW, HW]  B=1前提
        attn = attn[0]                        # [num_heads, HW, HW]
        num_heads, HW, _ = attn.shape
        H = W = int(HW ** 0.5)               # 正方形のfeature mapを仮定

        # 全headを平均 → [HW, HW]
        attn_mean = attn.mean(dim=0)          # [HW, HW]

        # CLSトークンがある場合はスキップ; ここではclass token不使用のViTを仮定
        # 各query位置からの平均attention（各位置がどこを見ているか）
        attn_map = attn_mean.mean(dim=0)      # [HW]  各key位置への平均attention
        attn_map = attn_map.reshape(H, W)     # [H, W]

        # [0,1]に正規化
        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(original_image)
        ax.imshow(
            attn_map,
            extent=[0, img_w, img_h, 0],     # 画像座標系に合わせる
            alpha=0.5,
            cmap='jet',
            interpolation='bilinear',
        )
        ax.set_title(f"Encoder Global Attention: {name}", fontsize=12)
        ax.axis('off')
        plt.colorbar(
            plt.cm.ScalarMappable(cmap='jet'),
            ax=ax, fraction=0.03, pad=0.04
        )
        save_path = output_dir / f"attn_enc_{name}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Encoder Attn] saved: {save_path}")


# ==============================================================
# (3) 追加: Decoder deformable cross attention mapの可視化
# 各decoder層のreference pointとattention weightを描画
#
# 前提: modelのforward()が以下を返すように改変済みであること
#   outputs['dec_attn_info']: list of dict（decoder層ごと）
#     各dictは以下を含む:
#       'ref_points':     [B, num_queries, num_points, 2]  (x,y 正規化座標)
#       'attn_weights':   [B, num_queries, num_heads, num_points]
# ==============================================================
def visualize_decoder_attention(image_path, dec_attn_info, output_dir):
    """
    dec_attn_info: list of dict（decoder層ごと）
      各dict:
        'ref_points':         [B, Q, n_levels, 4]   正規化cx,cy,w,h
        'attn_weights':       [B, Q, n_heads, n_levels*n_points]
        'sampling_locations': [B, Q, n_heads, n_levels, n_points, 2]  正規化xy
    """
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size

    for layer_idx, info in enumerate(dec_attn_info):
        # B=1前提でバッチ次元を除去
        ref_points         = info['ref_points'][0]         # [Q, n_levels, 4]
        attn_weights       = info['attn_weights'][0]       # [Q, n_heads, n_levels*n_points]
        sampling_locations = info['sampling_locations'][0] # [Q, n_heads, n_levels, n_points, 2]

        Q, n_heads, n_levels, n_points, _ = sampling_locations.shape

        # -------------------------------------------------------
        # attention weightで重み付けしたsampling locationのheatmap
        # attn_weights: [Q, n_heads, n_levels*n_points]
        #   → [Q, n_heads, n_levels, n_points] にreshape
        # -------------------------------------------------------
        attn_w = attn_weights.reshape(Q, n_heads, n_levels, n_points)
        # head平均 → [Q, n_levels, n_points]
        attn_w = attn_w.mean(dim=1).cpu().numpy()
        sampling_locs = sampling_locations.mean(dim=1).cpu().numpy()  # [Q, n_levels, n_points, 2]

        heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        r = 8
        for q in range(Q):
            for lvl in range(n_levels):
                for p in range(n_points):
                    x = int(np.clip(sampling_locs[q, lvl, p, 0] * img_w, 0, img_w - 1))
                    y = int(np.clip(sampling_locs[q, lvl, p, 1] * img_h, 0, img_h - 1))
                    w = attn_w[q, lvl, p]
                    y0, y1 = max(0, y - r), min(img_h, y + r + 1)
                    x0, x1 = max(0, x - r), min(img_w, x + r + 1)
                    heatmap[y0:y1, x0:x1] += w

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左: attention weight重み付きheatmap
        axes[0].imshow(original_image)
        axes[0].imshow(heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
        axes[0].set_title(f"Dec Layer {layer_idx+1}: Sampling Points (weighted by attn)")
        axes[0].axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                     ax=axes[0], fraction=0.03, pad=0.04)

        # 右: reference pointの散布図（level別に色分け）
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

    # preprocess（変更なし）
    image, orig_image_size = preprocess_image(args.input)
    image = image.to(device)
    orig_image_size = orig_image_size.to(device)

    images = nested_tensor_from_tensor_list([image])
    orig_image_sizes = torch.stack([orig_image_size])

    # ==============================================================
    # forward
    # 旧: outputs = model(images)
    # 新: attention情報も取得（model側の改変が必要、下記NOTE参照）
    # ==============================================================
    with torch.no_grad():
        outputs = model(images)

    # ==============================================================
    # (2) 修正: クラス別top-Kのbbox取得
    # 旧: postprocessors['bbox']で flatten top-100 を使用していた
    # ==============================================================
    result_per_class = get_topk_per_class(outputs, orig_image_sizes, k=TOP_K)

    # (2) top-Kスコアをprint
    print_topk_scores(result_per_class, k=TOP_K)

    # (2) 順位ごとに1枚ずつ描画・保存（合計TOP_K枚）
    visualize_topk_detections(args.input, result_per_class, args.output_dir, k=TOP_K)

    # ==============================================================
    # (3) Encoder global attention mapの可視化
    # NOTE: outputs['enc_attn_weights']はmodel側の改変で追加する必要がある
    #       改変不要ならこのブロックをコメントアウト
    # ==============================================================
    if 'enc_attn_weights' in outputs:
        visualize_encoder_attention(
            image_path=args.input,
            enc_attn_weights_list=outputs['enc_attn_weights'],  # list of [B, heads, HW, HW]
            output_dir=args.output_dir,
            layer_names=['enc_global_layer1', 'enc_global_layer3', 'enc_global_layer5'],
        )
    else:
        print("[INFO] enc_attn_weights not in outputs. Skip encoder attention visualization.")

    # ==============================================================
    # (3) Decoder deformable cross attention mapの可視化
    # NOTE: outputs['dec_attn_info']はmodel側の改変で追加する必要がある
    # ==============================================================
    if 'dec_attn_info' in outputs:
        visualize_decoder_attention(
            image_path=args.input,
            dec_attn_info=outputs['dec_attn_info'],
            output_dir=args.output_dir,
        )
    else:
        print("[INFO] dec_attn_info not in outputs. Skip decoder attention visualization.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR infer script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

####_init change from .lwdetr_demo import build
####TransformerDecoderLayer.forward_post()
def forward_post(self, tgt, memory,
                 tgt_mask=None, memory_mask=None,
                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                 pos=None, query_pos=None, query_sine_embed=None,
                 is_first=False, reference_points=None,
                 spatial_shapes=None, level_start_index=None):
    bs, num_queries, _ = tgt.shape

    # Self-Attention (no changed)
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
    # from: tgt2 = self.cross_attn(...)
    # to: tgt2, attn_weights, sampling_locations = self.cross_attn(...)
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
    # from: return tgt
    # to: return tgt, reference_points, attn_weights, sampling_locations
    # ==============================================================
    return tgt, reference_points, attn_weights, sampling_locations


def forward(self, tgt, memory, ...):
    # ==============================================================
    no changed
    # ==============================================================
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask,
                             pos, query_pos, query_sine_embed, is_first,
                             reference_points, spatial_shapes, level_start_index)

####TransformerDecoder.forward()
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

###Transformer.forward()
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

###LWDETR.forward()
def forward(self, samples, targets=None):
    features, poss, enc_attn_weights = self.backbone(samples)

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
    # only inference
    # ==============================================================

    
    if not self.training:
        out['enc_attn_weights'] = enc_attn_weights
        out['dec_attn_info'] = dec_attn_info

    return out

### vit.py
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_cae=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.use_cae = use_cae
        if use_cae:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None, return_attn=0):
        B, N, C = x.shape
        # qkv with shape (B, H, W, 3C)
        if self.use_cae:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        # pytorch naive implementation
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            attn.masked_fill_(mask.reshape(B, 1, 1, N).expand_as(attn), float('-inf'))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if return_attn:
            return x, attn

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window=False,
        use_cae=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_cae=use_cae,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window = window

        # for cae
        self.use_cae = use_cae
        if use_cae:
            init_values = 0.1
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, mask=None):
        """ Transformer Block forward"""
        B, HW, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if not self.window:
            x = x.reshape(B // 16, 16 * HW, C)
            if mask is not None:
                mask = mask.reshape(B // 16, 16 * HW)
        
        if self.window:
            return_attn = 0
        else:
            return_attn = 1
        
        global_attn = []

        if self.use_cae:
            x, attn = self.attn(x, mask, return_attn)
            x = self.gamma_1 * x
        else:
            x, attn = self.attn(x, mask, return_attn)
        global_attn.append(attn)

        if not self.window:
            x = x.reshape(B, HW, C)
            if mask is not None:
                mask = mask.reshape(B, HW)

        x = shortcut + self.drop_path(x)
        if self.use_cae:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x