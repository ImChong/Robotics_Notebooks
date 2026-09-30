# GAE-GeometricAutoEncoder（Tencent ARC）

> 来源归档

- **标题：** GAE: Learning a Geometry-Native Latent Space for 3D-Consistent World Generation
- **类型：** repo
- **链接：** <https://github.com/TencentARC/GAE-GeometricAutoEncoder>
- **论文：** <https://arxiv.org/abs/2609.24981>
- **项目页：** <https://jiah-cloud.github.io/GAE.github.io/>
- **默认 HF 权重：** <https://huggingface.co/TencentARC/GAE-D64-1B>
- **机构：** HKUST；Tencent ARC Lab (IEG)；HKU；UT Austin
- **许可：** Tencent 自定义许可（`LICENSE.txt`）— **仅学术用途**
- **入库日期：** 2026-09-30
- **一句话说明：** 两阶段 **geometry-native codec + conditional flow**：冻结 DA3 编解码几何，中间学 64/128 通道 latent；Stage2 在标准化 latent 上做 x-prediction flow matching，联合生成 RGB / depth / pose / point cloud。
- **沉淀到 wiki：** [`wiki/entities/paper-gae-geometry-native-autoencoder.md`](../../wiki/entities/paper-gae-geometry-native-autoencoder.md)

---

## 核心定位

公开论文代码：`gae/` API + `scripts/demo|train|eval|data` + `configs/`。内部研究文件名见 `docs/CODEBASE.md` / `docs/METHOD.md`，跑 demo 不必读。

**环境：** Python 3.10–3.12，torch 2.5.1；**不支持 3.13**。`pip install -e .` 即可（无需手设 `PYTHONPATH`）。

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| **一键 demo** | `bash scripts/demo/run_demo.sh`（无 venv 时在本地盘建 env）；`--smoke` 17 views / 25 steps |
| **I2V + 点云** | `python scripts/demo/generate.py --image ... --prompt-file ... --hf-repo TencentARC/GAE-D64-1B --total-views 81` |
| **Camera Studio** | `provision_fast_demo.sh` + `run_fast_demo.sh`（默认 8 GPU resident）；交互轨迹 + 同步 RGB/渐进 3D |
| **Gradio** | `app.py` Space；与 preset 轨迹 demo 并存 |
| **训练** | `scripts/train/` — Stage1 codec、Stage2 flow（见 README Training & evaluation） |
| **评测** | FVD / FID / 3D-consistency / MEt3R 等脚本于 `scripts/eval/` |
| **文档** | `docs/METHOD.md` 对齐论文 Eq. 8–14 |

---

## Stage 1 / Stage 2（与论文 §3 对齐）

```
images ──frozen DA3 encoder──> 4-level features ──normalize+concat──> X
X ──GAECodec.encode──> z (64 or 128 ch)
z ──GAECodec.decode──> rebuilt levels ──frozen DPT head──> depth / rays / pointmap
z ──learned RGB head──> RGB
```

- Codec 损失含 **L_tok**（C-RADIOv2.5-B 对齐）与 **L_struct**（DINOv2 patch 相似度，作用在 raw posterior mean）。
- Flow：**clean reference latents**（证据、非演化态）、**metric Plücker ray maps**、**text** cross-attention；仅 target-view latents 参与 ODE 积分。

---

## 交叉链接

- 项目页：[`sources/sites/gae-github-io.md`](../sites/gae-github-io.md)
- 论文摘录：[`sources/papers/gae_arxiv_2609_24981.md`](../papers/gae_arxiv_2609_24981.md)
