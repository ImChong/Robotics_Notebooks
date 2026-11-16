# THAW-VLA: Think Like a World Model, Act Like a VLA（arXiv:2609.24682）

> 来源归档（ingest）

- **标题：** THAW-VLA: Think Like a World Model, Act Like a VLA — Distilling World-Model Representations into Compact Robot Policies
- **类型：** paper / vla / world-model / representation-distillation / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.24682>
- **PDF：** <https://arxiv.org/pdf/2609.24682v3>
- **HTML 全文：** <https://arxiv.org/html/2609.24682v3>
- **核对范围：** 已通读 v3 摘要、引言、相关工作、方法（含式 1–3）、LIBERO / RoboCasa-GR1 / 真机实验、对照消融、结论及参考文献；并对照项目页与官方代码入口。
- **通读：** 已完成
- **复核日期：** 2026-11-12
- **项目页：** <https://thaw-vla.trung-dt.com/> — 归档见 [`sources/sites/thaw-vla-trung-dt-com.md`](../sites/thaw-vla-trung-dt-com.md)
- **代码：** **已开源** — <https://github.com/trungdt880/THAW-VLA>（MIT，vendored StarVLA）；归档见 [`sources/repos/thaw-vla.md`](../repos/thaw-vla.md)
- **权重：** Hugging Face collection [`termanteus/thaw-vla`](https://huggingface.co/collections/termanteus/thaw-vla) — **已公开**（2026-11-12 匿名可打开模型页与 Files 列表，其中可见 `final_model.pt`；README 仍留有过时 `private` 字样，实际下载及推理未在本次核验）
- **机构：** 威斯康星大学麦迪逊分校（University of Wisconsin–Madison）、伊利诺伊大学厄巴纳-香槟分校（University of Illinois Urbana-Champaign）— Trung Dao、Sankalp Yamsani、Jaden Park、Joohyung Kim、Yong Jae Lee
- **入库日期：** 2026-09-23
- **一句话说明：** 在标准 VLA 训练上加 **一条 cosine 特征对齐项**：冻结世界模型（Cosmos3-Nano）对训练帧 **预计算并缓存** teacher 特征，学生 QwenGR00T（0.8B）对齐后 **丢弃 projector**，部署图与未蒸馏 baseline **完全相同**（RTX 5090 上 32 ms / 1.86 GB）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://thaw-vla.trung-dt.com/> | LIBERO / RoboCasa-GR1 / 真机表、teacher 与 layer 消融 |
| GitHub | <https://github.com/trungdt880/THAW-VLA> | `00_setup` → `01_precompute` → `02_train` → `03/04_eval` |
| StarVLA | <https://github.com/starVLA/starVLA> | 学生框架与数据格式 |
| Cosmos 3 | 论文引用 Cosmos3-Nano | 默认 teacher，LM layer 24，4096-d |
| REPA | 表征对齐灵感 | 扩散训练中 hint 对齐；本文 teacher 为 WAM 特征 |

## 摘要级要点

- **问题：** VLA 只学 obs→action，缺少「世界如何响应动作」的目标，鲁棒性受数据覆盖上限；WAM 有该目标但 **秒级推理** 无法进控制环。
- **核心观察：** 世界模型的物理 grounding 已在 **内部特征** 中；生成未来只是产生这些特征的训练目标，可 **只蒸馏特征、不带生成器**。
- **方法：** $\mathcal{L} = \mathcal{L}_{\mathrm{act}} + 0.5\,\mathcal{L}_{\mathrm{align}}$；teacher 冻结、特征 **离线缓存**；训练时不加载 teacher；推理时去掉 alignment head。
- **学生：** QwenGR00T = Qwen3.5-VL 0.8B + flow-matching action expert（StarVLA）；4 flow steps。
- **Teacher：** Cosmos3-Nano 8B，layer-24 image tokens，每视角 mean-pool；LIBERO 2 cam → target_dim 8192，GR1 1 cam → 4096。

## 核心摘录（面向 wiki 编译）

### 1)  headline 数字

| 基准 | 无蒸馏 0.8B | THAW-VLA 0.8B | 备注 |
|------|-------------|---------------|------|
| LIBERO 四套件均值 | 95.3% | **97.9%** | 仿真 4 run 均值 |
| RoboCasa-GR1（24 env） | 48.2% | **50.5%** | 距 4B QwenGR00T 54.8% 约 4.3 pt |
| 真机三任务均值 | 56.7% | **66.7%** | AgileX Nero + TRIP-Bag；30 trials/cell |

### 2) Teacher 消融（LIBERO）

| Teacher | LIBERO avg |
|---------|------------|
| none | 95.3% |
| V-JEPA2-AC | 96.5% |
| Fast-WAM | 96.9% |
| Cosmos3-Nano | **97.9%** |

### 3) 部署成本（论文 Fig.1 / 项目页）

- THAW-VLA 0.8B：**32 ms**，**1.86 GB**（RTX 5090）
- 对照 DreamZero：~3 s，45.9 GB（H100 优化后仍远高于 VLA）

### 4) 开源状态（项目页 + GitHub，2026-09-23）

| 组件 | 状态 |
|------|------|
| 训练 / 评测 / deployment 代码 | **已开源**（MIT） |
| Teacher 预计算脚本 | `tools/cosmos3_precompute_targets.py` |
| 对齐模块 | `starVLA/model/modules/distill/fastwam_repa.py` |
| 发布 checkpoint | HF 模型页与文件列表可匿名访问（2026-11-12）；尚未下载运行，README 仍描述为 private |
| Cosmos3-Nano teacher 权重 | 需自备 ~33 GB checkpoint |

## v3 方法与评测核验补充

- **动作监督来源：** 示范动作；teacher 只提供视觉中间特征，不提供动作 label。
- **分视角特征：** student 对 image token mean-pool，经双层 MLP 映射后使用 `1 − cos` 对齐 teacher stop-grad 特征；总损失 `L_act + 0.5 L_align`。
- **评测协议：** LIBERO 约 27.3 万示范步，每任务 50 episode；RoboCasa-GR1 24 环境、训练共 2.4 万轨迹和约 602 万步；主表每环境测试 20 episode，2 seeds × 2 GPU 共四轮。
- **真机：** 三个任务分别为单臂水果、单臂鸡蛋、双臂水果交接；每个 policy/task 30 trials，对照 0.8B 83.3/46.7/40.0%，蒸馏后 93.3/60.0/46.7%。
- **消融限制：** RoboCasa backbone/layer 消融为单次 A100，layer sweep 只训完整日程的 1/4；不能与主表混比。
- **复现提示：** GitHub README 的 GR1 评测脚本标注每环境 50 次，与论文主表 20 次不同；比较结果要注明协议。

## 对 wiki 的映射

- 新建：[paper-thaw-vla](../../wiki/entities/paper-thaw-vla.md)
- 交叉：[vla](../../wiki/methods/vla.md)、[world-action-models](../../wiki/concepts/world-action-models.md)、[star-vla](../../wiki/methods/star-vla.md)、[paper-gift-intermediate-feature-training](../../wiki/entities/paper-gift-intermediate-feature-training.md)

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub 核查
- [x] 开源状态：代码已开源；HF checkpoint 文件列表公开，README 私有状态描述待同步
- [x] 源码运行时序图（对齐 README 四阶段脚本）
