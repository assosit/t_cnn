# Step 1
# From
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# To
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)          # マルチGPU全てにseedを反映
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = False     # 入力サイズごとのアルゴリズム自動選択を止める
torch.backends.cudnn.deterministic = True  # cuDNNの決定的アルゴリズムを強制

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # cuBLAS(行列積)の決定性確保
torch.use_deterministic_algorithms(True, warn_only=True)

# Step 2
# main.py のどこか(main関数の外)に追加
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# main()内、seed設定後に追加
g = torch.Generator()
g.manual_seed(seed)

# From
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                             drop_last=False, collate_fn=utils.collate_fn, 
                             num_workers=args.num_workers)

# To
# 修正後
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=args.num_workers,
                               worker_init_fn=seed_worker, generator=g)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                             drop_last=False, collate_fn=utils.collate_fn, 
                             num_workers=args.num_workers,
                             worker_init_fn=seed_worker, generator=g)








LW-DETR のViTエンコーダに「パッチ枝刈り機能」を追加実装してください。
仕様は添付の `LW-DETR_patch_pruning_spec.md` を正としてください。実装対象は主に以下です。

1. ViT backbone 側:
   - **window attention を全廃**し、全6ブロックを global attention に統一する
     （既存の window partition / reverse コードは削除、または dead code とせず
     完全に経路から外すこと）。
   - Patch Importance Scorer（Linear+Sigmoid、config で Conv2d+Sigmoid にも切替可能）を、
     `prune_block_indices`（デフォルト `[1, 3, 5]`）で指定されたブロックの**直前**に挿入する。
   - top-k によるハード選択で、各ステージ固定の keep_rate（デフォルト各0.5、累積
     1→1/2→1/4→1/8）分のトークンのみ残す（gather）。
   - 元のH0×W0グリッド上でのflat index（orig_pos_idx）を、pruning適用ごとに一貫して
     引き継ぎ、かつ**各ステージ後に昇順ソート**して次段へ渡す実装にする（3.2節参照）。
   - Block2, Block4, Block6 の出力をそれぞれ `feat_2`, `feat_4`, `feat_6` としてキャッシュする。
   - Block6直前の pruning stage で得られた `idx_final` を基準に、`torch.searchsorted` を用いて
     `feat_2`, `feat_4` から `idx_final` に対応する部分のみを抽出する（3.2節参照）。
     このとき `idx_final ⊂ idx_4 ⊂ idx_2` のネスト構造が保証されていることを前提としてよいが、
     念のためユニットテストで検証すること（5節の受け入れ基準3参照）。
   - `feat_2_sub`, `feat_4_sub`, `feat_6` の3系統それぞれについて、専用の学習可能パラメータ
     `mask_token_2` / `mask_token_4` / `mask_token_6`（config で共有にも切替可能）を用いて
     フル解像度 `(bs, N0, C)` へ scatter-back する（3.4節参照）。
   - 3系統をチャネル方向に concat → `(bs, N0, 3C)` → `(bs, 3C, H0, W0)` に reshape し、
     既存の C2f モジュールへそのまま入力する。C2f 以降（multi-scale pyramid 生成、
     `Transformer.forward` への `srcs`/`pos_embeds` 受け渡し、`transformer.py` 内のロジック）は
     完全に無改修とする。
   - `use_patch_pruning=False` のときは既存の forward（window attention含む旧経路）と
     完全に数値等価になること（回帰テストを書くこと）。

2. 損失関数側:
   - GT bbox（血清上端・下端の2 bbox）から、各pruningステージの解像度に対応する
     patch grid 上のターゲットマップ（パッチと bbox が少しでも重なれば1、
     それ以外0の二値、論理和で2bbox分をまとめる）を生成する関数を実装する。
   - 各ステージのスコアラー出力（ロジット）と、その時点で現存するトークンの
     `orig_pos_idx` を使ってgatherしたターゲットとの間で
     `BCEWithLogitsLoss(pos_weight=...)` を計算する（config で focal loss にも
     切替可能にする）。
   - 既存の detection loss（criterion.py 等、SetCriterion 相当のクラス）に
     `loss_prune` として統合し、`lambda_prune` で重み付けして合算する。

3. 学習まわり:
   - keep_rate のカリキュラムスケジューリング（epoch経過で1.0から目標値へ線形/コサイン
     で低下）をオプション実装する。config で on/off。
   - 各pruningステージのスコアマップを可視化するデバッグ用ユーティリティ
     （画像上に保持/削除パッチをオーバーレイ表示）を用意する。

4. テスト:
   - orig_pos_idx の gather → scatter が可逆であることを確認するユニットテスト
     （ダミーデータでpositionをそのまま特徴量に埋め込み、scatter後に復元されるか）。
   - use_patch_pruning=False 時の既存モデル（window attention含む旧経路）との出力一致テスト。
   - 各pruning stage後のトークン数が `N0*1/2 → N0*1/4 → N0*1/8` の期待通りであることの
     shapeテスト。
   - `idx_final ⊂ idx_4 ⊂ idx_2` のネスト構造が常に成立することを確認するテスト。
   - `torch.searchsorted` による feat_2/feat_4 からの抽出が正しい特徴を取得できているかの
     一致検証テスト。
   - `full_2`, `full_4`, `full_6` の scatter 結果について、`idx_final` に該当する位置は
     実特徴、それ以外は対応する mask_token と完全一致していることを確認するテスト。

既存コード（backbone, transformer.py, criterion.py 等）の該当ファイルをまず提示するので、
それを踏まえて具体的な差分（新規ファイル/既存ファイルへのパッチ）として実装してください。
仕様書中の Config パラメータ一覧に沿って、すべてハイパーパラメータとして外出しできる
ようにしてください。