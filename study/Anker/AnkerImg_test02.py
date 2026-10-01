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


# ４．表示
import matplotlib.pyplot as plt

# --------------------------------------------------
# ※※※※ 表示したい開始位置を指定（例: 0なら0~9枚目、10なら10~19枚目） ※※※※
start_idx = 0  # 0, 10, 20 など 10の倍数で切り替えると便利です
num_display = 10  # 一度に表示する枚数
# --------------------------------------------------

pred = predictions[0]
batch_size = len(pred.image)

# 表示範囲の指定（バッチサイズを超えないように制限）
end_idx = min(start_idx + num_display, batch_size)
actual_count = end_idx - start_idx

if actual_count <= 0:
    print(f"指定された start_idx ({start_idx}) はデータ範囲外です。(Total: {batch_size})")
else:
    # 10枚分（行数: actual_count, 列数: 2）の描画領域を作成
    fig, axes = plt.subplots(actual_count, 2, figsize=(10, 4 * actual_count))
    
    # 1枚だけ表示する場合の axes の次元調整
    if actual_count == 1:
        axes = [axes]

    for offset, i in enumerate(range(start_idx, end_idx)):
        # 画像とヒートマップの抽出
        image = pred.image[i].cpu().numpy().transpose(1, 2, 0)
        anomaly_map = pred.anomaly_map[i].cpu().numpy()
        pred_score = pred.pred_score[i].item()
        is_anomaly = pred.pred_label[i].item()

        # 正規化
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())

        # 次元調整
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze(0)

        ax_img = axes[offset][0]
        ax_map = axes[offset][1]

        # 左側：入力画像と判定結果
        ax_img.imshow(image)
        label_text = 'NG (Anomaly)' if is_anomaly else 'OK (Normal)'
        ax_img.set_title(f"[{i}] Input Image | Result: {label_text}")
        ax_img.axis("off")

        # 右側：ヒートマップ重畳
        ax_map.imshow(image)
        im = ax_map.imshow(anomaly_map, cmap="jet", alpha=0.4)
        ax_map.set_title(f"[{i}] Heatmap (Score: {pred_score:.4f})")
        ax_map.axis("off")

        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


pred = predictions[1]

# --- OK（正常: pred_label == 0）と判定された画像のインデックスを取得 ---
ok_indices = (pred.pred_label == 0).nonzero(as_tuple=True)[0].cpu().numpy()

# ページ切り替え用設定（例: OK画像の中の 0~9枚目 を表示）
start_ok_idx = 0  # 10枚以上OK画像がある場合、10, 20... でページ切り替え
num_display = 10

# 表示対象のインデックスを切り出し
target_indices = ok_indices[start_ok_idx : start_ok_idx + num_display]
actual_count = len(target_indices)

print(f"全 {len(pred.pred_label)} 枚中、OK（正常）と判定された画像: {len(ok_indices)} 枚")

if actual_count == 0:
    print("表示対象の OK 画像はありません。")
else:
    # 描画領域の作成
    fig, axes = plt.subplots(actual_count, 2, figsize=(10, 4 * actual_count))
    
    # 1枚だけ表示する場合の axes 次元の修正
    if actual_count == 1:
        axes = [axes]

    for offset, i in enumerate(target_indices):
        # データ抽出
        image = pred.image[i].cpu().numpy().transpose(1, 2, 0)
        anomaly_map = pred.anomaly_map[i].cpu().numpy()
        pred_score = pred.pred_score[i].item()

        # 正規化 & 次元調整
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze(0)

        ax_img = axes[offset][0]
        ax_map = axes[offset][1]

        # 左側：入力画像
        ax_img.imshow(image)
        ax_img.set_title(f"[{i}] Input Image | Result: OK (Normal)")
        ax_img.axis("off")

        # 右側：ヒートマップ重畳
        ax_map.imshow(image)
        im = ax_map.imshow(anomaly_map, cmap="jet", alpha=0.4)
        ax_map.set_title(f"[{i}] Heatmap (Score: {pred_score:.4f})")
        ax_map.axis("off")

        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
