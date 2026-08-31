# RTX 5090 (32GB, Blackwell sm_120) での TensorRT VAE エンジンビルド手順

ローカルの RTX 5060 Ti (16GB) では ONNX 生成・ビルドが遅い / 爆発する場合、
RTX 5090 (32GB, sm_120) をクラウドで借りて実行する手順。

**互換性**: RTX 5090 は RTX 5060 Ti と同じ Blackwell consumer (sm_120)。
5090 でビルドした .rtxplan は 5060 Ti で動作する（TensorRT バージョン一致前提）。

---

## ステップ 1: クラウド（RTX 5090）で ONNX 生成（GPU トレース・最速）

185f×512×512 のトレースは VRAM ~25GB → 5090 の 32GB に収まる。

```bash
# 1) リポジトリをクラウドへ
git clone <your-repo>  # または zip アップロード

# 2) VAE モデルを配置（ローカルからアップロード）
#    models/SEEDVR2/ema_vae_fp16.safetensors を配置

# 3) 依存インストール
pip install -r tools/cloud_requirements.txt

# 4) ONNX 生成（GPU トレース）
python tools/cloud_export_gpu.py --repo . --kind encoder --frames 185 \
    --output vae_encoder_185f_tile512.onnx --model ema_vae_fp16.safetensors

# デコーダも必要なら
python tools/cloud_export_gpu.py --repo . --kind decoder --frames 185 \
    --output vae_decoder_tile_256_185f.onnx --model ema_vae_fp16.safetensors
```

成功: `WORKER-OK encoder 185f -> ...`（数秒〜数十秒）

---

## ステップ 2: エンジンビルド（同じ 5090 上で）

```bash
python tools/cloud_build_engine.py vae_encoder_185f_tile512.onnx \
    --output vae_encoder_185f_tile512.rtxplan --workspace-gb 24

python tools/cloud_build_engine.py vae_decoder_tile_256_185f.onnx \
    --output vae_decoder_tile_256_185f.rtxplan --workspace-gb 24
```

成功: `OK: ...rtxplan (... MiB in ...s)`

---

## ステップ 3: ローカル（5060 Ti）へ配置

生成物をローカルにダウンロードして配置:

```
D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\seedvr2_videoupscaler\tensorrt_backend\artifacts\
  ├─ vae_encoder_185f_tile512.rtxplan
  ├─ vae_encoder_185f_tile512.onnx      （任意: 再ビルド用）
  ├─ vae_decoder_tile_256_185f.rtxplan
  └─ vae_decoder_tile_256_185f.onnx     （任意）
```

配置後、ComfyUI で Queue 実行 → エンジンが検出され 1 発実行される。

---

## 代替: ローカル CPU で ONNX 生成 → クラウド 5090 でビルドのみ

ONNX は GPU 非依存なので、ローカルで生成してクラウドへアップロードも可。

```bash
# ローカル（CPU トレース、遅い）
python tools/export_onnx_worker.py --repo D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\seedvr2_videoupscaler \
    --kind encoder --frames 185 --output vae_encoder_185f_tile512.onnx

# クラウド 5090 でビルドのみ
python tools/cloud_build_engine.py vae_encoder_185f_tile512.onnx \
    --output vae_encoder_185f_tile512.rtxplan --workspace-gb 24
```

---

## 注意

- **エンジンは GPU アーキテクチャ固有**。5090 (sm_120) 以外（A100/H100 等）でビルドした
  エンジンは 5060 Ti では動かない。必ず Blackwell (sm_120) でビルドすること。
- TensorRT バージョンはローカルと一致させる（`tensorrt-rtx==1.6.1.120`）。
- フレーム数（185/205/101 等）を変える場合は、ONNX とエンジンのファイル名の
  `185f` 部分を対応する値に変更すること。
