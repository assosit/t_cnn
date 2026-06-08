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
            """
            qs = q_indices if q_indices is not None else list(range(Q))
            all_w = np.array([attn_w_np[q, lvl, p]
                               for q in qs
                               for lvl in range(n_levels)
                               for p in range(n_points)])
            w_min, w_max = all_w.min(), all_w.max()
            w_range = w_max - w_min + 1e-8

            for q in qs:
                for lvl in range(n_levels):
                    for p in range(n_points):
                        x = sampling_np[q, lvl, p, 0] * img_w
                        y = sampling_np[q, lvl, p, 1] * img_h
                        w = attn_w_np[q, lvl, p]
                        color = WEIGHT_CMAP((w - w_min) / w_range)
                        ax.scatter(x, y, s=point_size, color=color,
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
            半径r=8の矩形領域にweightを加算して正規化。
            """
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
            r = 8
            qs = q_indices if q_indices is not None else list(range(Q))
            for q in qs:
                for lvl in range(n_levels):
                    for p in range(n_points):
                        x = int(np.clip(sampling_np[q, lvl, p, 0] * img_w, 0, img_w - 1))
                        y = int(np.clip(sampling_np[q, lvl, p, 1] * img_h, 0, img_h - 1))
                        w = attn_w_np[q, lvl, p]
                        y0_h, y1_h = max(0, y - r), min(img_h, y + r + 1)
                        x0_h, x1_h = max(0, x - r), min(img_w, x + r + 1)
                        heatmap[y0_h:y1_h, x0_h:x1_h] += w
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