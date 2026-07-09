def compute_cy_recall(outputs, targets, postprocessors,
                      thresholds=[7, 15, 25],
                      symmetric=False):
    orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, orig_sizes)

    stats = {f'recall_{cls}_{th}': []
             for cls in ['top', 'bot']
             for th in thresholds}

    for pred, target in zip(results, targets):
        pred_boxes  = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']

        gt_boxes  = target['boxes']
        gt_labels = target['labels']
        orig_h, orig_w = target['orig_size']

        gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes_xyxy = gt_boxes_xyxy * torch.tensor(
            [orig_w, orig_h, orig_w, orig_h],
            device=gt_boxes.device, dtype=gt_boxes.dtype
        )

        for gt_idx in range(len(gt_labels)):
            gt_label = gt_labels[gt_idx].item()
            gt_box   = gt_boxes_xyxy[gt_idx]
            gt_cy_px = (gt_box[1] + gt_box[3]) / 2

            cls_name = 'top' if gt_label == 0 else 'bot'

            cls_mask = (pred_labels == gt_label)
            if cls_mask.sum() == 0:
                for th in thresholds:
                    stats[f'recall_{cls_name}_{th}'].append(0.0)
                continue

            best_idx   = pred_scores[cls_mask].argmax()
            pred_cy_px = (pred_boxes[cls_mask][best_idx][[1, 3]].sum() / 2)

            # GT - pred に修正
            delta = gt_cy_px - pred_cy_px

            for th in thresholds:
                if symmetric:
                    tp = (-th <= delta <= th)
                else:
                    if gt_label == 0:  # 上端: δ∈[-th, 0] がTP
                        tp = (-th <= delta <= 0)
                    else:              # 下端: δ∈[0, th] がTP
                        tp = (0 <= delta <= th)
                stats[f'recall_{cls_name}_{th}'].append(float(tp))

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
    result['recall_mean'] = float(np.mean([
        result[f'recall_{cls}_{th}']
        for cls in ['top', 'bot']
        for th in thresholds
    ]))
    return result



def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    model.eval()
    ...
    # 既存の評価ループ
    all_outputs = []
    all_targets = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        outputs = model(samples)

        # 既存のcoco_evaluator処理
        ...

        # カスタム評価用に収集
        all_outputs.append(outputs)
        all_targets.extend(targets)

    # カスタムRecall計算
    # outputs を結合
    combined_outputs = {
        'pred_logits': torch.cat([o['pred_logits'] for o in all_outputs], dim=0),
        'pred_boxes':  torch.cat([o['pred_boxes']  for o in all_outputs], dim=0),
    }
    cy_stats = compute_cy_recall(combined_outputs, all_targets, postprocessors)
    
    # test_statsに追加
    test_stats.update({f'cy_{k}': v for k, v in cy_stats.items()})

    return test_stats, coco_evaluator

test_stats, coco_evaluator = evaluate(
    model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
)

# 従来のGIoUベースmAP
map_regular = test_stats['coco_eval_bbox'][0]

# 新: cy誤差ベースRecall (高いほど良い)
cy_recall = test_stats['cy_recall_mean']

# best model更新をcy_recallベースに変更
_isbest = best_map_holder.update(cy_recall, epoch, is_ema=False)

# ログ出力
print(f"mAP: {map_regular:.4f} | cy_recall_mean: {cy_recall:.4f}")
print(f"  top  7px: {test_stats['cy_recall_top_7']:.4f}  "
      f" bot  7px: {test_stats['cy_recall_bot_7']:.4f}")
print(f"  top 15px: {test_stats['cy_recall_top_15']:.4f}  "
      f" bot 15px: {test_stats['cy_recall_bot_15']:.4f}")
print(f"  top 25px: {test_stats['cy_recall_top_25']:.4f}  "
      f" bot 25px: {test_stats['cy_recall_bot_25']:.4f}")