def visualize_decoder_attention(image_path, dec_attn_info_list, output_dir,
                                 result_per_class=None):
    output_dir = Path(output_dir)
    original_image = Image.open(image_path).convert("RGB")
    img_w, img_h = original_image.size

    # ==============================================================
    # 上位クエリのindexを収集
    # {cat_id: [q_idx_top1, q_idx_top2, q_idx_top3]}
    # ==============================================================
    TOP_QUERIES = 3
    topk_query_idx = {}  # {cat_id: list of query_idx}
    if result_per_class is not None:
        for cat_id, res_info in result_per_class.items():
            topk_query_idx[cat_id] = \
                res_info['query_idx'][:TOP_QUERIES].numpy()

    for layer_idx, info in enumerate(dec_attn_info_list):
        ref_points         = info['ref_points'][0]         # [Q, n_levels, 4]
        attn_weights       = info['attn_weights'][0]       # [Q, n_heads, n_levels*n_points]
        sampling_locations = info['sampling_locations'][0] # [Q, n_heads, n_levels, n_points, 2]

        Q, n_heads, n_levels, n_points, _ = sampling_locations.shape

        attn_w        = attn_weights.reshape(Q, n_heads, n_levels, n_points).mean(dim=1)  # [Q, n_levels, n_points]
        sampling_locs = sampling_locations.mean(dim=1)                                     # [Q, n_levels, n_points, 2]
        attn_w_np     = attn_w.cpu().numpy()
        sampling_np   = sampling_locs.cpu().numpy()
        ref_np        = ref_points.cpu().numpy()

        # --------------------------------------------------------------
        # heatmapとscatterを生成する共通関数
        # query_indices=Noneのとき全クエリ、指定時はそのクエリのみ
        # query_labels: {q_idx: label_str} — Noneのときラベル非表示
        # --------------------------------------------------------------
        def make_sampling_heatmap(query_indices=None):
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
            r = 8
            qs = query_indices if query_indices is not None else range(Q)
            for q in qs:
                for lvl in range(n_levels):
                    for p in range(n_points):
                        x = int(np.clip(sampling_np[q, lvl, p, 0] * img_w, 0, img_w - 1))
                        y = int(np.clip(sampling_np[q, lvl, p, 1] * img_h, 0, img_h - 1))
                        w = attn_w_np[q, lvl, p]
                        y0, y1 = max(0, y - r), min(img_h, y + r + 1)
                        x0, x1 = max(0, x - r), min(img_w, x + r + 1)
                        heatmap[y0:y1, x0:x1] += w
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        def draw_sampling_scatter(ax, query_indices=None, query_labels=None):
            """
            query_labels: {q_idx: label_str}
                各クエリの代表点（全levelの重み付き重心）にラベルを描画する。
                Noneのときラベル非表示。
            """
            colors_level = ['red', 'blue', 'green', 'orange']
            qs = query_indices if query_indices is not None else range(Q)
            for lvl in range(n_levels):
                xs, ys = [], []
                for q in qs:
                    for p in range(n_points):
                        xs.append(sampling_np[q, lvl, p, 0] * img_w)
                        ys.append(sampling_np[q, lvl, p, 1] * img_h)
                ax.scatter(xs, ys, s=5 if query_indices is None else 30,
                           alpha=0.4,
                           color=colors_level[lvl % len(colors_level)],
                           label=f"level {lvl}")
            ax.legend(loc='upper right', fontsize=8)

            # クエリごとのラベル描画（指定クエリのみ）
            if query_labels and query_indices is not None:
                for q in qs:
                    if q not in query_labels:
                        continue
                    # 全level・全pointの重み付き重心を代表点とする
                    total_w = 0.0
                    cx, cy = 0.0, 0.0
                    for lvl in range(n_levels):
                        for p in range(n_points):
                            w = attn_w_np[q, lvl, p]
                            cx += sampling_np[q, lvl, p, 0] * img_w * w
                            cy += sampling_np[q, lvl, p, 1] * img_h * w
                            total_w += w
                    if total_w > 0:
                        cx /= total_w
                        cy /= total_w
                    ax.annotate(
                        query_labels[q],
                        xy=(cx, cy),
                        fontsize=8,
                        color='white',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6),
                        ha='center', va='bottom',
                    )

        def draw_ref_scatter(ax, query_indices=None, query_labels=None):
            """
            query_labels: {q_idx: label_str}
                各クエリの代表点（level0のcx,cyを使用）にラベルを描画する。
                Noneのときラベル非表示。
            """
            colors_level = ['red', 'blue', 'green', 'orange']
            qs = query_indices if query_indices is not None else range(Q)
            for lvl in range(n_levels):
                cx = [ref_np[q, lvl, 0] * img_w for q in qs]
                cy = [ref_np[q, lvl, 1] * img_h for q in qs]
                ax.scatter(cx, cy, s=5 if query_indices is None else 30,
                           alpha=0.4,
                           color=colors_level[lvl % len(colors_level)],
                           label=f"level {lvl}")
            ax.legend(loc='upper right', fontsize=8)

            # クエリごとのラベル描画（指定クエリのみ）
            if query_labels and query_indices is not None:
                for q in qs:
                    if q not in query_labels:
                        continue
                    # level 0 の cx, cy を代表点とする
                    px = ref_np[q, 0, 0] * img_w
                    py = ref_np[q, 0, 1] * img_h
                    ax.annotate(
                        query_labels[q],
                        xy=(px, py),
                        fontsize=8,
                        color='white',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6),
                        ha='center', va='bottom',
                    )

        # ==============================================================
        # 描画対象の定義
        # (タイトルサフィックス, query_indices, query_labels)
        #
        # query_labels: {q_idx: label_str}
        #   同一query_idが複数クラスにまたがる場合はクラス名を連結し、
        #   重複するquery_idは1つにまとめる（重複排除）。
        # ==============================================================
        targets = [('All Queries', None, None)]
        if topk_query_idx:
            for rank in range(TOP_QUERIES):
                # query_id をキーにして重複を排除しながらラベルを集約する
                # {q_idx: [label_part, ...]}
                q_label_map = {}
                for cat_id, idxs in topk_query_idx.items():
                    if rank < len(idxs):
                        q_idx = int(idxs[rank])
                        class_name = CLASSES.get(cat_id, str(cat_id))
                        label_part = f"{class_name}\nq={q_idx}"
                        if q_idx not in q_label_map:
                            q_label_map[q_idx] = []
                        q_label_map[q_idx].append(label_part)

                if not q_label_map:
                    continue

                # 重複クエリのラベルはクラス名を改行で連結
                query_indices = list(q_label_map.keys())
                query_labels  = {q: '\n'.join(parts) for q, parts in q_label_map.items()}

                # タイトル用サフィックス（重複クエリはまとめて表示）
                label_parts = []
                for q, parts in q_label_map.items():
                    label_parts.append(f"q={q} ({'/'.join(p.split(chr(10))[0] for p in parts)})")
                title_suffix = f"Top-{rank+1} ({', '.join(label_parts)})"

                targets.append((title_suffix, query_indices, query_labels))

        # ==============================================================
        # Sampling point: 全クエリ・top1・top2・top3 の4枚
        # ==============================================================
        for title_suffix, q_indices, q_labels in targets:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 左: heatmap
            heatmap = make_sampling_heatmap(q_indices)
            axes[0].imshow(original_image)
            im = axes[0].imshow(heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
            axes[0].set_title(f"Sampling Heatmap\n{title_suffix}")
            axes[0].axis('off')
            fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.04)

            # 右: scatter（ラベル付き）
            axes[1].imshow(original_image)
            draw_sampling_scatter(axes[1], q_indices, q_labels)
            axes[1].set_title(f"Sampling Scatter\n{title_suffix}")
            axes[1].axis('off')

            fig.suptitle(f"Dec Layer {layer_idx+1}: Sampling Points", fontsize=13)
            plt.tight_layout()

            # ファイル名のサフィックス
            fname_suffix = title_suffix.split(' ')[0].lower()  # 'all' or 'top-1' etc.
            save_path = output_dir / f"attn_dec_layer{layer_idx+1}_sampling_{fname_suffix}.jpg"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"[Decoder Attn] saved: {save_path}")

        # ==============================================================
        # Reference point: 全クエリ・top1・top2・top3 の4枚
        # ==============================================================
        for title_suffix, q_indices, q_labels in targets:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(original_image)
            draw_ref_scatter(ax, q_indices, q_labels)
            ax.set_title(
                f"Dec Layer {layer_idx+1}: Reference Points\n{title_suffix}",
                fontsize=12
            )
            ax.axis('off')
            plt.tight_layout()

            fname_suffix = title_suffix.split(' ')[0].lower()
            save_path = output_dir / f"attn_dec_layer{layer_idx+1}_refpoint_{fname_suffix}.jpg"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"[Decoder Attn] saved: {save_path}")