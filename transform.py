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