from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim

# 自作データセットのロード設定
datamodule = Folder(
    name="mylocal",
    root="./datasets/mylocal/Anker_bat",
    normal_dir="train/good",         # 学習用の正常画像フォルダ
    test_split_mode="from_dir",       # testフォルダから読み込む指定
    normal_test_dir="test/good",      # テスト用の正常画像フォルダ
    abnormal_dir="test/defect_A",# テスト用の異常画像フォルダ
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4,
)

# 1. モデルとエンジンの構築
model = Padim()
engine = Engine(
    max_epochs=1,
    default_root_dir="./results/mylocal/Anker_bat",
    accelerator="auto",
)

# 2. 学習（特徴抽出）
engine.fit(model=model, datamodule=datamodule)

# 3. 推論
predictions = engine.predict(model=model, datamodule=datamodule)

i = 0

# 5. 可視化
pred = predictions[0]

# [0] を指定して1件目のデータを抽出
image = pred.image[i].cpu().numpy().transpose(1, 2, 0)
anomaly_map = pred.anomaly_map[i].cpu().numpy()

# pred_score も [0] を指定してスカラー値を取得
pred_score = pred.pred_score[i].item()

# 画像の表示範囲を [0, 1] に正規化（描画崩れ防止）
if image.max() > 1.0 or image.min() < 0.0:
    image = (image - image.min()) / (image.max() - image.min())

# anomaly_map が 3次元(1, H, W)になっている場合の対策
if anomaly_map.ndim == 3:
    anomaly_map = anomaly_map.squeeze(0)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image)
axes[0].set_title("Input Image")
axes[0].axis("off")

im = axes[1].imshow(anomaly_map, cmap="jet")
axes[1].set_title(f"Anomaly Map (Score: {pred_score:.4f})")
axes[1].axis("off")

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# 1件目の判定結果を出力
is_anomaly = pred.pred_label[i].item()
print(f"判定結果: {'異常 (Anomaly)' if is_anomaly else '正常 (Normal)'}")
