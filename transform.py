class LongSideResize(object):
    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        w_orig, h_orig = img.size

        if w_orig >= h_orig:
            new_w = size
            new_h = int(round(h_orig * size / w_orig))
        else:
            new_h = size
            new_w = int(round(w_orig * size / h_orig))
        
        rescaled_img = F.resize(img, (new_h, new_w))
        w, h = rescaled_img.size

        if target is None:
            return rescaled_img, None

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
        ratio_width, ratio_height = ratios

        target = target.copy()

        # 2. boxをスケール
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height],
            )
            target["boxes"] = scaled_boxes

        # 3. areaをスケール
        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        # 4. 変換後画像サイズ
        target["size"] = torch.tensor([h, w])
        target["pad_size"] = torch.tensor([H_pad, W_pad])

        return rescaled_img, target

class LongSideResize(object):
    """
    長辺を基準にアスペクト比を保ってリサイズ。
    正方形paddingは行わない。
    """
    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img, target=None):
        # 1. 長辺基準でリサイズ
        long_side = random.choice(self.sizes)
        w_orig, h_orig = img.size  # PIL: (width, height)

        scale = long_side / max(w_orig, h_orig)
        new_w = int(round(w_orig * scale))
        new_h = int(round(h_orig * scale))

        resized_img = F.resize(img, (new_h, new_w))  # F.resize は (h, w)

        if target is None:
            return resized_img, None

        target = target.copy()

        ratio_w = new_w / w_orig
        ratio_h = new_h / h_orig

        # 2. boxをスケール
        if "boxes" in target:
            boxes = target["boxes"]
            target["boxes"] = boxes * torch.as_tensor(
                [ratio_w, ratio_h, ratio_w, ratio_h],
                dtype=boxes.dtype,
                device=boxes.device,
            )

        # 3. areaをスケール
        if "area" in target:
            target["area"] = target["area"] * (ratio_w * ratio_h)

        # 4. 変換後画像サイズ
        target["size"] = torch.tensor([new_h, new_w])

        # 5. maskを画像と同じサイズにリサイズ
        if "masks" in target:
            target["masks"] = interpolate(
                target["masks"][:, None].float(),
                size=(new_h, new_w),
                mode="nearest",
            )[:, 0] > 0.5

        return resized_img, target
    


if args.pretrain_weights is not None:
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
        # add support to exclude_keys
        # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
        if args.pretrain_exclude_keys is not None:
            assert isinstance(args.pretrain_exclude_keys, list)
            for exclude_key in args.pretrain_exclude_keys:
                checkpoint['model'].pop(exclude_key)
        if args.pretrain_keys_modify_to_load is not None:
            assert isinstance(args.pretrain_keys_modify_to_load, list)
            for modify_key_to_load in args.pretrain_keys_modify_to_load:
                if modify_key_to_load in checkpoint['model']:
                    del checkpoint['model'][modify_key_to_load]
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

if args.pretrain_weights is not None:
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # ---------------------------------------------------------------
    # 1. pretrain_exclude_keys: 明示的に除外するキーを削除
    # ---------------------------------------------------------------
    if args.pretrain_exclude_keys is not None:
        assert isinstance(args.pretrain_exclude_keys, list)
        for exclude_key in args.pretrain_exclude_keys:
            if exclude_key in checkpoint_model:
                print(f"[Pretrain] Excluding key: {exclude_key}")
                checkpoint_model.pop(exclude_key)

    # ---------------------------------------------------------------
    # 2. pretrain_keys_modify_to_load: obj365→coco変換（必要な場合）
    # ---------------------------------------------------------------
    if args.pretrain_keys_modify_to_load is not None:
        from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
        assert isinstance(args.pretrain_keys_modify_to_load, list)
        for modify_key_to_load in args.pretrain_keys_modify_to_load:
            if modify_key_to_load in checkpoint_model:
                checkpoint_model[modify_key_to_load] = get_coco_pretrain_from_obj365(
                    model_without_ddp.state_dict()[modify_key_to_load],
                    checkpoint_model[modify_key_to_load]
                )

    # ---------------------------------------------------------------
    # 3. shape不一致キーの自動検出と部分転用
    #    COCO(81次元) → 自前データ(3次元) のhead重みを処理
    # ---------------------------------------------------------------
    current_model_state = model_without_ddp.state_dict()
    shape_mismatched_keys = []
    keys_to_remove = []

    for key in list(checkpoint_model.keys()):
        if key not in current_model_state:
            # モデルに存在しないキーはスキップ（strict=Falseで対処済みだが明示）
            print(f"[Pretrain] Key not in current model, skipping: {key}")
            keys_to_remove.append(key)
            continue

        ckpt_shape = checkpoint_model[key].shape
        model_shape = current_model_state[key].shape

        if ckpt_shape != model_shape:
            shape_mismatched_keys.append(key)
            print(f"[Pretrain] Shape mismatch - {key}: "
                  f"checkpoint={ckpt_shape}, model={model_shape}")

            # --- class_embed (分類head) の部分転用 ---
            # COCOの重みのうち、background(index=0)の重みのみ転用し、
            # クラス固有の重みは現モデルの初期値を保持する。
            # （background特徴の汎用性は高いため転用価値がある）
            if 'class_embed' in key:
                new_param = current_model_state[key].clone()  # モデルの初期値をベースに
                src = checkpoint_model[key]  # COCO: shape [81, d] or [81]

                # weight: [num_classes, d_model]  /  bias: [num_classes]
                # background(index=0)の重みはドメイン不変性が高いため転用
                min_classes = min(new_param.shape[0], src.shape[0])
                # index 0 (background) のみ転用
                new_param[0] = src[0]
                print(f"[Pretrain] Partially loaded class_embed (background only): {key}")
                checkpoint_model[key] = new_param

            # --- bbox_embed (回帰head) の部分転用 ---
            # bbox回帰は座標予測でありクラス非依存な場合が多い。
            # 次元が合う範囲でできるだけ転用する。
            elif 'bbox_embed' in key:
                new_param = current_model_state[key].clone()
                src = checkpoint_model[key]
                # 共通次元をスライスして転用（shape違いは先頭次元のみと仮定）
                min_dim0 = min(new_param.shape[0], src.shape[0])
                if new_param.dim() == 1:
                    new_param[:min_dim0] = src[:min_dim0]
                elif new_param.dim() == 2:
                    min_dim1 = min(new_param.shape[1], src.shape[1])
                    new_param[:min_dim0, :min_dim1] = src[:min_dim0, :min_dim1]
                print(f"[Pretrain] Partially loaded bbox_embed: {key}")
                checkpoint_model[key] = new_param

            else:
                # 上記以外のshape不一致キーは除外（strict=Falseで安全にスキップ）
                print(f"[Pretrain] Skipping mismatched key: {key}")
                keys_to_remove.append(key)

    # 除外対象キーをcheckpointから削除
    for key in keys_to_remove:
        checkpoint_model.pop(key, None)

    # ---------------------------------------------------------------
    # 4. ロード実行（strict=False: 不足キーは初期値のまま）
    # ---------------------------------------------------------------
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
        checkpoint_model, strict=False
    )
    print(f"[Pretrain] Missing keys (使用初期値): {missing_keys}")
    print(f"[Pretrain] Unexpected keys (無視): {unexpected_keys}")
    print(f"[Pretrain] Shape mismatch keys (部分転用 or スキップ): {shape_mismatched_keys}")

    # ---------------------------------------------------------------
    # 5. EMAモデルの再構築
    # ---------------------------------------------------------------
    if args.use_ema:
        del ema_m
        ema_m = ModelEma(model_without_ddp)
        print("[Pretrain] EMA model re-initialized from loaded weights.")


### ver 2
if args.pretrain_weights is not None:
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # ---------------------------------------------------------------
    # 1. pretrain_exclude_keys: 明示的に除外するキーを削除
    # ---------------------------------------------------------------
    if args.pretrain_exclude_keys is not None:
        assert isinstance(args.pretrain_exclude_keys, list)
        for exclude_key in args.pretrain_exclude_keys:
            if exclude_key in checkpoint_model:
                print(f"[Pretrain] Excluding key: {exclude_key}")
                checkpoint_model.pop(exclude_key)

    # ---------------------------------------------------------------
    # 2. pretrain_keys_modify_to_load: obj365→coco変換（必要な場合）
    # ---------------------------------------------------------------
    if args.pretrain_keys_modify_to_load is not None:
        from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
        assert isinstance(args.pretrain_keys_modify_to_load, list)
        for modify_key_to_load in args.pretrain_keys_modify_to_load:
            if modify_key_to_load in checkpoint_model:
                checkpoint_model[modify_key_to_load] = get_coco_pretrain_from_obj365(
                    model_without_ddp.state_dict()[modify_key_to_load],
                    checkpoint_model[modify_key_to_load]
                )

    # ---------------------------------------------------------------
    # 3. shape不一致・モデルに存在しないキーを自動除外
    #    → 除外されたキーはモデルの初期値がそのまま使われる
    # ---------------------------------------------------------------
    current_model_state = model_without_ddp.state_dict()
    keys_to_remove = []

    for key in list(checkpoint_model.keys()):
        if key not in current_model_state:
            keys_to_remove.append(key)
        elif checkpoint_model[key].shape != current_model_state[key].shape:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        print(f"[Pretrain] Skipping key (not in model or shape mismatch): {key}")
        checkpoint_model.pop(key)

    # ---------------------------------------------------------------
    # 4. ロード実行
    # ---------------------------------------------------------------
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
        checkpoint_model, strict=False
    )
    print(f"[Pretrain] Missing keys (初期値を使用): {missing_keys}")
    print(f"[Pretrain] Skipped keys (初期値を使用): {keys_to_remove}")

    # ---------------------------------------------------------------
    # 5. EMAモデルの再構築
    # ---------------------------------------------------------------
    if args.use_ema:
        del ema_m
        ema_m = ModelEma(model_without_ddp)