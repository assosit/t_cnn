def draw_sampling_scatter(ax, query_indices=None):
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

    # ==============================================================
    # 追加: 上位クエリの場合のみクラス名ラベルを描画
    # 各クエリの全sampling pointの重心位置にラベルを表示
    # ==============================================================
    if query_indices is not None and topk_query_idx:
        for cat_id, idxs in topk_query_idx.items():
            for rank, q_idx in enumerate(idxs):
                if q_idx not in query_indices:
                    continue
                # 全level・全pointの重心を計算
                xs_q = [sampling_np[q_idx, lvl, p, 0] * img_w
                        for lvl in range(n_levels) for p in range(n_points)]
                ys_q = [sampling_np[q_idx, lvl, p, 1] * img_h
                        for lvl in range(n_levels) for p in range(n_points)]
                cx = np.mean(xs_q)
                cy = np.mean(ys_q)
                label_text = f"{CLASSES.get(cat_id, str(cat_id))} (q={q_idx})"
                ax.text(cx, cy - 12, label_text,
                        color=CLASS_COLORS.get(cat_id, 'white'),
                        fontsize=9, fontweight='bold',
                        ha='center',
                        bbox=dict(facecolor='black', alpha=0.5, pad=2))
    ax.legend(loc='upper right', fontsize=8)


def draw_ref_scatter(ax, query_indices=None):
    colors_level = ['red', 'blue', 'green', 'orange']
    qs = query_indices if query_indices is not None else range(Q)
    for lvl in range(n_levels):
        cx = [ref_np[q, lvl, 0] * img_w for q in qs]
        cy = [ref_np[q, lvl, 1] * img_h for q in qs]
        ax.scatter(cx, cy, s=5 if query_indices is None else 30,
                   alpha=0.4,
                   color=colors_level[lvl % len(colors_level)],
                   label=f"level {lvl}")

    # ==============================================================
    # 追加: 上位クエリの場合のみクラス名ラベルを描画
    # 各クエリの全levelの重心位置にラベルを表示
    # ==============================================================
    if query_indices is not None and topk_query_idx:
        for cat_id, idxs in topk_query_idx.items():
            for rank, q_idx in enumerate(idxs):
                if q_idx not in query_indices:
                    continue
                # 全levelの重心を計算
                xs_q = [ref_np[q_idx, lvl, 0] * img_w for lvl in range(n_levels)]
                ys_q = [ref_np[q_idx, lvl, 1] * img_h for lvl in range(n_levels)]
                cx = np.mean(xs_q)
                cy = np.mean(ys_q)
                label_text = f"{CLASSES.get(cat_id, str(cat_id))} (q={q_idx})"
                ax.text(cx, cy - 12, label_text,
                        color=CLASS_COLORS.get(cat_id, 'white'),
                        fontsize=9, fontweight='bold',
                        ha='center',
                        bbox=dict(facecolor='black', alpha=0.5, pad=2))
    ax.legend(loc='upper right', fontsize=8)