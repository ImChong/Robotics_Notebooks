# SSLA / SSLA-Det（haohq19/ssla）

> 来源归档

- **标题：** Low-Latency Event-Based Object Detection with Spatially-Sparse Linear Attention
- **类型：** repo / event-camera / object-detection / linear-attention / async
- **来源：** 清华大学 · 苏黎世大学 RPG · 上海科技大学（Haiqing Hao 等）
- **链接：** <https://github.com/haohq19/ssla>
- **论文：** <https://arxiv.org/abs/2603.06228>（ECCV 2026）
- **视频：** <https://youtu.be/qaVeSqEt8IM>
- **前置代码：** [haohq19/eva](https://github.com/haohq19/eva) — 见 [`eva.md`](eva.md)
- **入库日期：** 2026-09-12
- **一句话说明：** **SSLA-Det** 官方实现：MOS 线性注意力异步检测，`train.py` / `test.py` / `compute_flops.py`；Gen1 + N-Caltech101 配置与 Google Drive checkpoint。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md`](../../wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md)

---

## 核心定位

ECCV 2026 论文 *Low-Latency Event-Based Object Detection with Spatially-Sparse Linear Attention* 的官方代码：**Spatially-Sparse Linear Attention（SSLA）** + **Mixture-of-Spaces（MOS）** 骨干，面向 **逐事件低延迟目标检测**。

---

## 仓库入口

| 组件 | 说明 |
|------|------|
| 安装 | Python 3.12；PyTorch 2.6 + cu124；Lightning；torch-scatter；pycocotools；h5py |
| 数据 | Gen1：<https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/> → `./data/Gen1/{data,bbox}/{train,val,test}` |
| N-Caltech101 | 按 README 指向 [uzh-rpg/dagr](https://github.com/uzh-rpg/dagr) 预处理 → `./data` |
| 训练 | `python train.py --train_cfg=configs/train_cfg_gen1_{b,s,m,l}.yaml`（或 `train_cfg_ncaltech101_*.yaml`） |
| 评测 | `python test.py --test_cfg=configs/test_cfg_gen1_*.yaml` |
| FLOPs | `python compute_flops.py --model_cfg=configs/model/MOS_*.yaml --dataset={gen1,ncaltech101}` |
| 权重 | [Google Drive checkpoints](https://drive.google.com/drive/folders/1sCcDlYpDL_SXwxHUmaxwfj_vh_16wna9?usp=sharing) → `./checkpoints` |
| 模型规模 | `configs/model/MOS_{B,S,M,L}.yaml` 与对应 N-Caltech101 变体 |

---

## 关键模块

| 路径 | 说明 |
|------|------|
| `model_mos.py` | MOS 检测骨干 |
| `layers/mos_attention.py` | Mixture-of-spaces 注意力 |
| `layers/linear_attention.py` | 线性注意力算子 |
| `layers/async_sparse_module.py` | 异步稀疏模块 |
| `layers/yolox_head.py` | 检测头 |
| `dataset/gen1.py` / `ncaltech101.py` | 数据加载 |
| `ops/gla/` | 分块线性注意力内核（chunk / naive） |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-sa-2603-06228](../../wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md) | 论文实体与结论 |
| [eva.md](eva.md) | 同团队 ICLR 2026 A2S 前置；Gen1 检测强基线 |
| [paper-microsaccade-inspired-event-camera](../../wiki/entities/paper-microsaccade-inspired-event-camera.md) | 事件相机硬件与感知栈上下文 |
| [paper-simple-evrgb-cal](../../wiki/entities/paper-simple-evrgb-cal.md) | 事件—RGB 标定工具链 |
