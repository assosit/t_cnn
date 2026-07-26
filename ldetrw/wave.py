# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
#
# ============================================================================
# [Wave-ViT改修] 変更点サマリ
#   1. HaarDWT2D / HaarIDWT2D を追加（Wave-ViTのDWT/IDWTを外部ライブラリ非依存で軽量再実装）
#   2. WaveletAttentionWrapper を追加（既存 Attention を無改造のままラップ）
#   3. Block.__init__: self.attn を Attention -> WaveletAttentionWrapper(Attention(...)) に変更
#   4. Block.forward: 引数に h, w を追加し、そのまま self.attn へ伝搬するだけ
#   5. ViT.forward: blk(x, mask=None) -> blk(x, h, w, mask=None) に呼び出しを変更
#   depth / dim / window_block_indexes / MLP / LayerNorm / 残差構造は一切変更していません。
# ============================================================================

"""
ViT encoder
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, Mlp, trunc_normal_

from util.box_ops import box_cxcywh_to_xyxy


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


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

    def forward(self, x, mask=None):
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

        return x


# ============================================================================
# [追加] Haar-DWT / IDWT ヘルパーモジュール
#   Wave-ViTのDWT_Function/IDWT_Function（pywt + 独自autograd.Function）とは異なり、
#   Haarに固定し標準の F.conv2d / F.conv_transpose2d だけで実装する。
#   Haarの2x2基底は正規直交（各フィルタのL2ノルム=1）なので、
#   DWT(conv2d, stride=2) の「転置」を IDWT(conv_transpose2d, stride=2) に使えば
#   厳密に完全再構成（ロスレス）できる。外部ライブラリ非依存・ONNXエクスポートも
#   Conv2d/ConvTranspose2dのみで構成されるため問題なく対応可能。
# ============================================================================
class HaarDWT2D(nn.Module):
    """
    入力 (B, C, H, W) -> 
        x_ll  : (B, C, H/2, W/2)  低周波成分。Attentionにはこれだけを渡す。
        x_high: (x_lh, x_hl, x_hh) 各 (B, C, H/2, W/2)。IDWTで使うためキャッシュしておく。
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1., 1.], [1., 1.]]) * 0.5
        lh = torch.tensor([[1., 1.], [-1., -1.]]) * 0.5
        hl = torch.tensor([[1., -1.], [1., -1.]]) * 0.5
        hh = torch.tensor([[1., -1.], [-1., 1.]]) * 0.5
        # (4, 1, 2, 2)
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filt', filt, persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, \
            f"HaarDWT2D requires even H, W, got ({H}, {W})"

        # weight: (4C, 1, 2, 2)。repeat(C,...) で [ll,lh,hl,hh, ll,lh,hl,hh, ...] という
        # チャネル毎4連続ブロックの並びになり、groups=C の depthwise conv2d の期待する
        # 「入力チャネルiは出力チャネル[4i:4i+4]を担当」というレイアウトと一致する。
        weight = self.filt.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
        out = F.conv2d(x, weight, stride=2, groups=C)          # (B, 4C, H/2, W/2)
        out = out.reshape(B, C, 4, H // 2, W // 2)
        x_ll, x_lh, x_hl, x_hh = out.unbind(dim=2)
        return x_ll, (x_lh, x_hl, x_hh)


class HaarIDWT2D(nn.Module):
    """HaarDWT2Dの逆変換。ConvTranspose2dで LL+LH+HL+HH から元の解像度を完全再構成する。"""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1., 1.], [1., 1.]]) * 0.5
        lh = torch.tensor([[1., 1.], [-1., -1.]]) * 0.5
        hl = torch.tensor([[1., -1.], [1., -1.]]) * 0.5
        hh = torch.tensor([[1., -1.], [-1., 1.]]) * 0.5
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer('filt', filt, persistent=False)

    def forward(self, x_ll, x_high):
        x_lh, x_hl, x_hh = x_high
        B, C, H2, W2 = x_ll.shape

        # (B, C, 4, H2, W2) -> (B, 4C, H2, W2)  ※DWT側と同じチャネル並びに揃える
        x = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=2).reshape(B, C * 4, H2, W2)
        weight = self.filt.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)  # (4C, 1, 2, 2)
        out = F.conv_transpose2d(x, weight, stride=2, groups=C)  # (B, C, H2*2, W2*2)
        return out


# ============================================================================
# [追加] WaveletAttentionWrapper
#   LW-DETRの既存 Attention を無改造のままラップし、Attention計算の直前にDWTで
#   H/2×W/2へダウンサンプリング、直後にIDWTで元解像度へ復元する。
# ============================================================================
class WaveletAttentionWrapper(nn.Module):
    """
    重要な注意点（グローバルattentionブロックのタイル順について）:
        ViT.forward は特徴マップを最初から
            x.reshape(B, 4, h, 4, w, C).permute(0,1,3,2,4,5).reshape(B*16, h*w, C)
        という「4x4タイル分割」の形に変形して各Blockへ渡す。
        - window=True のブロック: 入力は (B*16, h*w, C)。これは1タイル = 真に連続した
          h×w の空間領域なので、そのまま (h, w) の2Dグリッドとして扱ってよい。
        - window=False のブロック: Block.forward内で (B, 16*h*w, C) にマージされるが、
          このマージは単純な reshape であり、16タイルを「タイル順（qy, qx走査）」で
          連結しただけで、真のラスタ（行優先）順の (4h, 4w) グリッドにはなっていない。
          DWTは真の空間隣接性が前提の演算なので、このケースでは
              タイル順 → 真の2Dグリッド → DWT → Attention → IDWT → タイル順に戻す
          という変換が必須。

    設計上の注意（Windowサイズについて）:
        DWT/IDWTは本ラッパー内、すなわちAttention呼び出しの直前直後だけで完結し、
        Blockを抜けた後は必ず元解像度(H/16)に戻る。したがって各window（タイル）が
        担当する実空間の面積自体は変化しない（間引かれるのはタイル内部のトークン数のみ）。
        よって「H/32相当で受容野が実質2倍になる」問題は本設計では発生せず、
        window_block_indexes や4分割という構造（=絶対制約）を変更する必要はない。
    """

    def __init__(self, attn: nn.Module, window: bool):
        super().__init__()
        self.attn = attn          # 元の LW-DETR Attention をそのまま保持（中身は無改造）
        self.window = window
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

    @staticmethod
    def _tiles_to_grid(x, h, w):
        """(B, 16*h*w, C) タイル順 -> (B, 4h, 4w, C) 真のラスタグリッド"""
        B, N, C = x.shape
        x = x.reshape(B, 4, 4, h, w, C)                 # (B, qy, qx, h, w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 4 * h, 4 * w, C)
        return x

    @staticmethod
    def _grid_to_tiles(x, h, w):
        """(B, 4h, 4w, C) 真のラスタグリッド -> (B, 16*h*w, C) タイル順（上の逆変換）"""
        B, H_img, W_img, C = x.shape
        x = x.reshape(B, 4, h, 4, w, C)                  # (B, qy, h, qx, w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 4 * 4 * h * w, C)
        return x

    def forward(self, x, h, w, mask=None):
        """
        x: (B, N, C)
            window=True  -> N == h*w      (1タイル分, 真の2D隣接性を持つ)
            window=False -> N == 16*h*w   (16タイル, タイル順)
        h, w: ViT.forward で一度だけ計算される基準タイルサイズ (= H_patch//4, W_patch//4)。
        """
        B, N, C = x.shape

        if self.window:
            H_cur, W_cur = h, w
            grid = x.reshape(B, H_cur, W_cur, C)
        else:
            H_cur, W_cur = 4 * h, 4 * w
            grid = self._tiles_to_grid(x, h, w)          # (B, H_cur, W_cur, C)

        assert H_cur % 2 == 0 and W_cur % 2 == 0, \
            f"WaveletAttentionWrapper: 偶数の空間サイズが必要 got ({H_cur}, {W_cur})"

        grid = grid.permute(0, 3, 1, 2).contiguous()      # (B, C, H_cur, W_cur)
        x_ll, x_high = self.dwt(grid)                     # x_ll: (B, C, H_cur/2, W_cur/2)
        Hh, Wh = x_ll.shape[-2:]

        x_ll_tok = x_ll.permute(0, 2, 3, 1).reshape(B, Hh * Wh, C)  # (B, N/4, C)

        if mask is not None:
            # 現状 LW-DETR は常に mask=None で呼び出しているため未使用経路だが、将来
            # paddingマスク等が渡された場合に備え、2x2 max-pool で保守的にダウンサンプル
            # する（4トークンのうち1つでもmask対象なら、縮小後トークンもmask対象とする）。
            mask_grid = mask.reshape(B, 1, H_cur, W_cur).float()
            mask_ds = F.max_pool2d(mask_grid, kernel_size=2, stride=2)
            mask_ds = mask_ds.reshape(B, Hh * Wh).bool()
        else:
            mask_ds = None

        # ここが「Attentionの実行」部分。LW-DETRのAttentionは完全に無改造のまま、
        # トークン数が N から N/4 に減った状態で呼ばれるだけ。
        x_ll_tok = self.attn(x_ll_tok, mask_ds)

        x_ll = x_ll_tok.reshape(B, Hh, Wh, C).permute(0, 3, 1, 2).contiguous()
        grid_out = self.idwt(x_ll, x_high)                 # (B, C, H_cur, W_cur) 高周波(エッジ)を復元
        grid_out = grid_out.permute(0, 2, 3, 1)             # (B, H_cur, W_cur, C)

        if self.window:
            out = grid_out.reshape(B, N, C)
        else:
            out = self._grid_to_tiles(grid_out, h, w)       # 元のタイル順へ戻す

        return out


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

        raw_attn = Attention(                     # ← 元のAttentionはそのまま生成（無改造）
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_cae=use_cae,
        )
        # ============ [変更点 1/2] Attentionをラップするだけ ============
        self.attn = WaveletAttentionWrapper(raw_attn, window=window)
        # =================================================================

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

    # ============ [変更点 2/2] forward の引数に h, w を追加し、そのまま self.attn へ渡す ============
    def forward(self, x, h, w, mask=None):
        """ Transformer Block forward"""
        B, HW, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if not self.window:
            x = x.reshape(B // 16, 16 * HW, C)
            if mask is not None:
                mask = mask.reshape(B // 16, 16 * HW)

        if self.use_cae:
            x = self.gamma_1 * self.attn(x, h, w, mask)
        else:
            x = self.attn(x, h, w, mask)

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
    # ====================================================================================


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature_indexes:list=None,
        use_cae=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window=True if i in window_block_indexes else False,
                use_cae=use_cae,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.window_block_indexes = window_block_indexes
        out_feature_indexes = [ind if ind >= 0 else ind + depth for ind in out_feature_indexes]
        out_feature_indexes = [ind for ind in range(depth) if ind in out_feature_indexes]

        self._out_features = [True if i in out_feature_indexes else False for i in range(depth)]
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)
        assert self._out_features[-1] is True

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

        self._export = False

    def export(self):
        self._export = True
        self.pos_embed_export = get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, (40, 40)
        ).detach()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self._export:
                x = x + self.pos_embed_export
            else:
                x = x + get_abs_pos(
                    self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
                )

        B, H, W, C = x.shape
        assert (H % 4 == 0) and (W % 4 == 0)
        h, w = H // 4, W // 4

        x = x.reshape(B, 4, h, 4, w, C).permute(
            0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)
        out = []
        for idx, blk in enumerate(self.blocks):
            # ============ [変更点] h, w を追加で渡すだけ ============
            x = blk(x, h, w, mask=None)
            # ==========================================================
            if self._out_features[idx]:
                out.append(x.reshape(B, 4, 4, h, w, C).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, C, H, W))
        return out



import re

def remap_legacy_vit_state_dict(state_dict: dict) -> dict:
    """
    WaveletAttentionWrapper導入前（Attentionを素で持っていた）チェックポイントのキーを、
    導入後のモジュール階層（Block.attn = WaveletAttentionWrapper(Attention)）に合わせて
    リマップする。

    変換対象: "blocks.<idx>.attn.<attn_param>" -> "blocks.<idx>.attn.attn.<attn_param>"
    対象パラメータ: qkv.weight / qkv.bias / proj.weight / proj.bias / q_bias / v_bias
    それ以外（norm1, norm2, mlp, gamma_1, gamma_2, patch_embed, pos_embed）は無変換。
    """
    pattern = re.compile(r"^(blocks\.\d+\.attn)\.(qkv|proj|q_bias|v_bias)")
    new_state_dict = {}
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            new_key = f"{m.group(1)}.attn.{m.group(2)}" + k[m.end():]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# 使用例
ckpt = torch.load("lwdetr_pretrained.pth", map_location="cpu")
state_dict = ckpt["model"] if "model" in ckpt else ckpt  # チェックポイント形式に合わせて調整
remapped = remap_legacy_vit_state_dict(state_dict)

missing, unexpected = model.load_state_dict(remapped, strict=False)
print("missing:", missing)       # dwt.filt / idwt.filt はバッファ(persistent=False)なので
                                  # そもそもstate_dictに現れず、ここには出てこないはず
print("unexpected:", unexpected) # 空であるべき