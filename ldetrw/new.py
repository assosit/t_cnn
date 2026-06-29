def loss_boxes(self, outputs, targets, indices, num_boxes):
    assert 'pred_boxes' in outputs
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    # --- 通常のL1損失 (cx, w, h成分) ---
    loss_bbox_base = F.l1_loss(src_boxes, target_boxes, reduction='none')

    # --- cy成分の非対称L1損失 ---
    # src_boxes, target_boxes は (cx, cy, w, h) フォーマット
    delta_cy = src_boxes[:, 1] - target_boxes[:, 1]  # pred_cy - gt_cy

    # 各boxのクラスラベルを取得
    target_labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    # 0: top (上端), 1: bottom (下端) を想定
    is_top = (target_labels == 0)   # 上端クラス
    is_bot = (target_labels == 1)   # 下端クラス

    alpha = self.asymmetric_alpha  # ペナルティ係数 (例: 3.0)

    # 上端: delta_cy < 0 (予測がGTより上) に厳しいペナルティ
    penalty_top = torch.where(
        delta_cy < 0,
        alpha * delta_cy.abs(),   # 厳しいペナルティ
        delta_cy.abs()            # 通常ペナルティ
    )

    # 下端: delta_cy > 0 (予測がGTより下) に厳しいペナルティ
    penalty_bot = torch.where(
        delta_cy > 0,
        alpha * delta_cy.abs(),   # 厳しいペナルティ
        delta_cy.abs()            # 通常ペナルティ
    )

    # cy損失を合成 (top/bot以外は通常L1のまま)
    loss_cy = torch.where(is_top, penalty_top,
              torch.where(is_bot, penalty_bot,
              delta_cy.abs()))  # 想定外クラスはフォールバック

    # cx, cy, w, h を結合 (cyだけ差し替え)
    loss_bbox = torch.stack([
        loss_bbox_base[:, 0],  # cx
        loss_cy,               # cy (非対称)
        loss_bbox_base[:, 2],  # w
        loss_bbox_base[:, 3],  # h
    ], dim=1)

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    return losses



## Matcherは任意

def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1, 
             focal_alpha=0.25,
             asymmetric_alpha=3.0):  # 追加
    ...
    self.asymmetric_alpha = asymmetric_alpha

@torch.no_grad()
def forward(self, outputs, targets, group_detr=1):
    ...
    # 既存コード
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [N, 4]
    tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [M, 4]
    tgt_ids  = torch.cat([v["labels"] for v in targets])  # [M]

    # cx, w, h は従来通りの対称L1
    # cy のみ非対称L1に変更
    
    # cx, w, h の L1コスト: [N, M]
    cost_bbox_cxwh = (
        torch.cdist(out_bbox[:, [0]], tgt_bbox[:, [0]], p=1) +  # cx
        torch.cdist(out_bbox[:, [2]], tgt_bbox[:, [2]], p=1) +  # w
        torch.cdist(out_bbox[:, [3]], tgt_bbox[:, [3]], p=1)    # h
    )

    # cy の非対称L1コスト: [N, M]
    delta_cy = out_bbox[:, 1:2] - tgt_bbox[:, 1:2].T  # [N, M]

    alpha_asym = self.asymmetric_alpha
    is_top = (tgt_ids == 0).unsqueeze(0)  # [1, M]
    is_bot = (tgt_ids == 1).unsqueeze(0)  # [1, M]

    # 上端: δ<0 に厳しいペナルティ
    cost_cy_top = torch.where(delta_cy < 0,
        alpha_asym * delta_cy.abs(),
        delta_cy.abs()
    )
    # 下端: δ>0 に厳しいペナルティ
    cost_cy_bot = torch.where(delta_cy > 0,
        alpha_asym * delta_cy.abs(),
        delta_cy.abs()
    )
    # 通常L1 (フォールバック)
    cost_cy_sym = delta_cy.abs()

    cost_cy = torch.where(is_top, cost_cy_top,
              torch.where(is_bot, cost_cy_bot,
              cost_cy_sym))

    cost_bbox = cost_bbox_cxwh + cost_cy

    # Final cost matrix (以降は変更なし)
    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou