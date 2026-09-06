# 指示
提供したLW-DETRのレポジトリを基に、モデルレベルのMixture of Experts (MoE) を実装したラッパークラス `DynamicModelMoE` をPyTorchで作成してください。検体容器の境界検出タスクを想定しており、推論時の高速化とロバスト性の両立が目的です。

# 要件
1. **ルーター構造**: 
   - 入力画像を受け取る軽量なCNNルーター（ResNet18の最初の数層、またはカスタムの軽量MobileNetV3ベース）を実装してください。
   - 出力次元は `num_experts` とし、各Expertへのロジットを出力します。

2. **Gumbel-SoftmaxとSTE**:
   - `F.gumbel_softmax(logits, tau, hard=True)` を使用してOne-hotなマスクを取得してください。

3. **バッチ処理と勾配接続（最重要）**:
   - バッチサイズが2以上の場合、全てのExpertに全バッチを入力するのは非効率です。
   - One-hotマスクを元に、バッチ内のデータを各Expertへ振り分け（インデックスによるスライス）、それぞれのExpertの順伝播を実行後、元のバッチ順序に再結合（scatter/結合）してください。
   - 再結合した出力テンソルに対して、Gumbel-Softmaxの出力マスクの対応する値を掛け合わせる (`out * y`) ことで、ルーターへ正しく勾配が逆流する計算グラフを構築してください。

4. **推論モード (eval) の最適化**:
   - `self.training == False` の場合はGumbel-Softmaxを使用せず、`argmax` を用いてバッチ内の各画像にTop-1のExpertのみを適用する効率的な条件分岐を実装してください。

5. **Load Balancing Loss**:
   - ルーターのロジット（Softmax適用後）と実際の選択頻度から、特定のExpertへの極端な偏りを防ぐ `Load Balancing Loss` を計算し、フォワードパスの戻り値（またはクラスのプロパティ）として返せるようにしてください。

# 構成例
```python
class DynamicModelMoE(nn.Module):
    def __init__(self, experts: List[nn.Module], num_classes: int):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = # 軽量CNNの実装
        self.tau = 1.0 # アニーリング対応可能に

    def forward(self, images, targets=None):
        # 実装をお願いします


################################PART2################################
# 指示
提供したLW-DETRのコードを基に、解像度が異なる2つのExpertモデルを動的にルーティングするMoEラッパークラス `MultiResDynamicMoE` をPyTorchで実装してください。

# 前提条件
- 対象タスク: 検体容器内の血清と分離剤の境界検出（バウンディングボックス出力、座標は0-1で正規化済）。
- 入力データ: 原画像 `images` (256x1536)。

# 要件
1. **Expertの構成**:
   - `expert_A`: 低解像度 (128x768) で動作する標準LW-DETR。
   - `expert_B`: 高解像度 (256x1536) で動作するパッチ枝刈り導入版LW-DETR（別途実装済みのクラスをインスタンス化して渡す想定）。
   
2. **ルーター構造**:
   - 128x768の画像を受け取り、2クラスのロジットを出力する軽量なCNN。

3. **解像度の動的生成と順伝播**:
   - `forward(images)` メソッド内で、まず `images` (256x1536) に対して `F.interpolate` を用いて低解像度画像 `images_low` (128x768) を生成してください。
   - ルーターには `images_low` を入力し、Gumbel-Softmax (hard=True) で選択マスク `y` を取得します。
   - 【学習時】: 勾配を流すため、バッチ内の各画像をインデックスで振り分けます。
     - `expert_A` が選ばれたサンプルには `images_low` の該当スライスを入力。
     - `expert_B` が選ばれたサンプルには `images` の該当スライスを入力。
     - 双方の出力を元のバッチ順に結合し、出力に `y` を掛け合わせる（`out * y`）ことでルーターに勾配を接続してください。
   - 【推論時 (eval)】: `argmax` を用いて1つのExpertだけを動かします。選ばれたExpertに合わせて `images_low` または `images` のどちらか一方だけを渡して推論を実行してください。

4. **Load Balancing Loss**:
   - ルーターのロジット出力から Load Balancing Loss を計算し、タスクのLossに加算できるようプロパティまたは戻り値として返してください。

# 構成例
```python
class MultiResDynamicMoE(nn.Module):
    def __init__(self, expert_A: nn.Module, expert_B: nn.Module):
        super().__init__()
        self.expert_A = expert_A
        self.expert_B = expert_B
        self.router = # 軽量CNN
        
    def forward(self, images, targets=None):
        # 1. 画像のダウンサンプル (images -> images_low)
        # 2. ルーターによるロジット計算 (入力は images_low)
        # 3. ルーティング処理 (学習時: 両方計算して勾配接続, 推論時: Top-1のみ計算)
        # 実装をお願いします