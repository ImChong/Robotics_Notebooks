# PredActor 评测模型工件（MasterYip/PredActor_Artifacts）

> 来源归档（Hugging Face model repository；最近复核：2026-10-08）

- **仓库：** <https://huggingface.co/MasterYip/PredActor_Artifacts>
- **类型：** model artifacts / evaluation checkpoints
- **对应项目：** PredActor（[论文](https://arxiv.org/abs/2609.24840) · [代码](https://github.com/MasterYip/PredActor)）
- **许可证：** 模型卡标注 MIT；第三方资产及依赖遵循各自许可
- **入库日期：** 2026-10-08
- **用途：** 为官方公开 MuJoCo evaluator 提供 PDP051 策略和配套 G1 MotionCLIP 语义编码器；不是训练数据集。

## 发布工件

| 路径 | 用途 | 大小 | SHA-256 |
|------|------|------:|---------|
| `checkpoints/predactor/pdp051/latest.ckpt` | 文本条件 PredActor PDP051 policy | 49,212,894 bytes | `2d963b32786f2989c6472726df9fcfe6b385590127e12e1f549a4b7d77488b2e` |
| `checkpoints/motionclip/g1-model-xyz-clip/checkpoint_0100.pth.tar` | G1 微调 MotionCLIP 语义编码器 | 542,749,069 bytes | `66a127df4958b346089b2020f2705c7456d9db0ee8b4bd9518608b708b35fc3c` |

仓库提供 `SHA256SUMS`。官方 `scripts/hf_download.py` 按 `scripts/hf_manifest.yaml` 下载并检查文件身份；推荐经该入口获取，不要执行来历不明的 pickle-compatible checkpoint。

## 可复现范围

- 与公开代码仓的 `uv sync --locked` 和 `predactor-eval` 配套，用于本地 MuJoCo simulation evaluation。
- 模型卡明确标注：训练数据、实验输出、硬件部署 bundle 和其它 checkpoint 未包含。
- `dataset/` 目录目前只是未来公开数据的预留位置，没有 dataset payload。
- MuJoCo evaluator 能验证 checkpoint/软件兼容性，不证明新场景的步态质量或真机安全。

## 对 wiki 的映射

- [PredActor 论文与项目实体](../../wiki/entities/paper-predactor.md)
- [官方评测代码仓归档](predactor.md)
- [项目页归档](../sites/predactor-masteryip-github-io.md)