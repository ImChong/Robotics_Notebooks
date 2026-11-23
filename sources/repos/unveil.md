# yuhuan-wu/UNVEIL（SAM 3 可提示概念图）

> 来源归档（官方仓库、源码与发布件核对）

- **类型：** repo / computer-vision / promptable-concept-maps
- **代码：** <https://github.com/yuhuan-wu/UNVEIL>
- **论文：** [arXiv:2611.25747](https://arxiv.org/abs/2611.25747)；[论文核对](../papers/unveil_pcm_arxiv_2611_25747.md)
- **开放程度：** **已开源**。仓库包含推理、训练、评测、数据准备代码，以及 v1.0 附加权重；基础 SAM 3 模型权重另受 Hugging Face gated access / 相应许可约束。
- **许可：** UNVEIL 仓库标注 MIT；第三方 SAM 3 权重与数据遵守各自许可。
- **附加权重：** [UNVEIL v1.0 safetensors](https://github.com/yuhuan-wu/UNVEIL/releases/download/v1.0/unveil_v1.safetensors)（约 138 MB）
- **入库日期：** 2026-11-21
- **核对范围：** 仓库 README 与 docs/skill.md；scripts/predict.py；unveil/__init__.py；unveil/model/{unveil,backbone,heads,sam3}.py；训练/评测 CLI 目录；未在本环境执行 CUDA 推理。

## 最短推理入口（官方 README / docs/skill.md）

```bash
git clone https://github.com/yuhuan-wu/UNVEIL.git
cd UNVEIL
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40"
pip install -e ".[eval,data,test]"
wget -P checkpoints https://github.com/yuhuan-wu/UNVEIL/releases/download/v1.0/unveil_v1.safetensors
python scripts/predict.py assets/examples/coco_000000228144.jpg --prompts mirror edges "salient object" person --overlay --out outputs/example
```

需 CUDA GPU、支持的 NVIDIA 驱动，以及已获许可的 `facebook/sam3` checkpoint（或本地 `sam3.pt`）。`--joint` 只允许 `salient object` 与 `camouflaged object` 配对；`--path sam3` 比较官方未改动的 SAM 3 语义路径；`--no-gate` 返回被 presence 判为「none」的原始概念图，供对照评测。

## 源码入口与数据流（静态核验）

| 路径 | 作用 |
|---|---|
| `scripts/predict.py` | 读取图片和短语 CLI 参数，调用 `unveil.load` / `model.predict`，写入 8-bit mask PNG、可选 overlay 及 `predictions.json` |
| `unveil/__init__.py` | 对外 lazy `load()`，转调 `unveil.model.unveil.load` |
| `unveil/model/unveil.py` | `load` 建立冻结 SAM 3 + UNVEIL checkpoint；`predict` 在 single/joint 分支产出上采样图与 presence，并按阈值置零 |
| `unveil/model/sam3.py` | 调 `build_sam3_image_model`，锁定 SAM 3 源码 commit；图像缩放到 1008×1008 并标准化 |
| `unveil/model/backbone.py` | `vit_prefix`、`image_features`、`pair_features` 组合共享视觉与文本提示特征 |
| `unveil/model/heads.py` | `ConceptHeads`：融合编码、语义像素解码与单独 presence 解码；presence memory/text 使用 stop-gradient |
| `scripts/train.py`、`scripts/evaluate.py` | 已公开训练/评测命令行入口，实验配置见 `configs/unveil.yaml` 与 README |
| `unveil/viz/render.py` | CLI 输出灰度 map 与图像叠加 |

具体过程是 `scripts/predict.py` → `unveil.load()` → `UNVEIL.predict()` → 图文特征 → semantic / presence 两个输出 → 门控 → mask / `predictions.json`。原生 `instances` 仍可通过 SAM 3 processor 取得实例掩码。

## 复现与风险边界

- 显存：官方使用说明标明单提示约 4.2 GiB、八提示约 6.6 GiB；图像编码一次可以配多个短语，但不能据此宣称 CPU 可运行。
- `facebook/sam3` 是 gated 模型；不要在日志、代码或笔记中保存 Hugging Face 访问 token。
- 原仓库提供预训练权重及评测入口；本次仅核实公开实现，未实测 GPU 速度与 PCM-Bench 分数。
- “none” 拒绝决策的可靠性随 prompt 的训练负例覆盖变化；对阴影、结构等类别应独立评估负样本误报。
- 单图像 2D 概念地图并非可直接执行的机器人检测、3D 定位或运动规划算法。

## 对 wiki 的映射

- [UNVEIL 独立论文详情](../../wiki/entities/paper-unveil-pcm-2611-25747.md)
- [SAM 3 基座](../../wiki/entities/paper-sam3.md)
