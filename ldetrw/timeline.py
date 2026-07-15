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