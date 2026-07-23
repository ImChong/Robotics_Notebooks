# EgoSteer: A Full-Stack System Towards Steerable Dexterous Manipulation from Egocentric Videos（arXiv:2607.09701）

> 来源归档（ingest）

- **标题：** EgoSteer: A Full-Stack System Towards Steerable Dexterous Manipulation from Egocentric Videos
- **缩写：** **EgoSteer**（系统总称）；管线 **EgoSmith**；控制侧 **Unified Robot Stack**
- **类型：** paper / vla / egocentric-video / dexterous-manipulation / world-model / dagger / teleoperation
- **arXiv：** <https://arxiv.org/abs/2607.09701>（PDF：<https://arxiv.org/pdf/2607.09701>）
- **项目页：** <https://egosteer.github.io/> — 归档见 [`sources/sites/egosteer-github-io.md`](../sites/egosteer-github-io.md)
- **代码：** **已开源（Apache-2.0）** — [`egosteer/egosteer`](https://github.com/egosteer/egosteer)、[`egosteer/egosmith`](https://github.com/egosteer/egosmith)、[`egosteer/robot-stack`](https://github.com/egosteer/robot-stack)；仓库归档见 [`sources/repos/egosteer.md`](../repos/egosteer.md)、[`sources/repos/egosmith.md`](../repos/egosmith.md)、[`sources/repos/egosteer-robot-stack.md`](../repos/egosteer-robot-stack.md)
- **权重：** HF [`EgoSteer/EgoSteer-3B-Base`](https://huggingface.co/EgoSteer/EgoSteer-3B-Base)、[`EgoSteer/EgoSteer-3B-RealMan`](https://huggingface.co/EgoSteer/EgoSteer-3B-RealMan)（Apache-2.0）
- **数据：** 项目页宣称将开源处理后 egocentric / 真机数据集；截至 **2026-07-23** HF 组织下 **datasets 仍为 0**（仅模型已发）
- **作者：** Yifan Zhong\*、Zhang Chen\*、Tianrui Guan\*、Fanlian Zeng\*、Yuyao Ye、Tianjia He、Ka Nam Lui、Jiayi Li、Tingrui Zhang、Ruilin Yan、Xinhao Ji、Guangyu Zhao、Wenjie Lou、Jiayuan Zhang、Yuanpei Chen†、Yaodong Yang†（\*共一；†通讯）
- **机构：** 北京大学人工智能研究院；PKU–PsiBot Joint Lab；宾夕法尼亚大学（UPenn，Zeng）
- **入库日期：** 2026-07-23
- **一句话说明：** 用 **EgoSmith** 把野外 egocentric 视频策展成 **9.6K h** 全标注预训练语料，经 **统一 Robot Stack**（遥操作 + HITL DAgger）与 **世界模型增强的 flow-VLA（EgoSteer）** 落地到双灵巧手，在 **40+** 自由语言任务上约 **75%** 成功率，并支持双具身长程 few-shot。

## 开源状态（步骤 2.5）

- **项目页核查（2026-07-23）：** Header 提供 Code / Data / Model 入口；GitHub org [`egosteer`](https://github.com/egosteer) 下三仓库均公开（Apache-2.0）；HF 已发 Base 与 RealMan 权重；**处理后大数据集页仍空**（项目页写「几周内 / 几天内开源」）。
- **结论：** **部分开源** — **代码 + 权重 + 示例数据已开源**；**9.6K h / 187 h 全量处理后数据集待发布**。

## 摘录 1：问题与主张（§1）

- **痛点：** generalist 策略的 **steerability**（自由语言跟随）在 **双灵巧手** 上稀缺；瓶颈是缺少大规模、语言对齐、动作准确的数据与可扩展全栈。
- **主张：** 三件套闭环——**EgoSmith（人视频策展）→ Robot Stack（遥操作 / 推理 / DAgger）→ EgoSteer（WM 增强 VLA）**。
- **动作接口：** 统一 **腕 SE(3) + 指尖关键点**；训练期 **RTC**；训练-only 世界专家在 **DINOv3** 潜空间预测动作诱导未来特征。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-egosteer.md`](../../wiki/entities/paper-egosteer.md)；与 [EgoScale](../../wiki/methods/egoscale.md)、[EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)、[VLA](../../wiki/methods/vla.md)、[DAgger](../../wiki/methods/dagger.md)、[Manipulation](../../wiki/tasks/manipulation.md) 互链。

## 摘录 2：EgoSmith + Robot Stack + EgoSteer（§3–§5）

- **EgoSmith 四阶段：** 预过滤（光流 locomotion 门 + YOLO 手门）→ 改进 4D（HaWoR + **DPVO** + **Any4D**，相对 HaWoR **~9×** 吞吐）→ 五级语言标注（Qwen3.5-VL）→ episode/chunk/frame 后过滤。
- **语料：** 12 个原始 egocentric 源 → **9.60K h / 2.09M episodes / 1.04B frames**。
- **Robot Stack：** SynGlove-Air + Vive；mink IK；相对运动映射 HITL（handover >85%）；**187 h / 193 任务** 真机数据 + **3 轮 DAgger（8.3 h）**。
- **EgoSteer：** Qwen3-VL-2B + DiT flow-matching 动作专家；CFM；训练-only WM expert 回归未来 DINOv3；HSDP + WebDataset，8×A800 上约 **44.5% MFU / 97 samples/s**。

**对 wiki 的映射：** 实体页画全栈 flowchart + 源码运行时序图（对齐 `train.py` / `run_server.sh` / robot-stack）。

## 摘录 3：实验要点（§6）

| 设定 | 关键数字（论文） |
|------|----------------|
| 40 任务自由语言（含 seen / compositional / unseen） | 总体约 **75%**；compositional **65%**；unseen **62%** |
| DAgger（4 难任务） | FT **22.5%** → DG **62.5%**（+8.3 h 纠偏） |
| 相对 π₀.₅ / Being-H0.5（10 较易任务） | **74%** vs **22%** / **39%** |
| 消融（1K h 预训练设定） | No WM **31%** / No RTC **39%** / Noisy data **33%** / Ours **44%** |
| Few-shot 长程 | RealMan 折盒 **75%**（120 demos）；AgiBot-G1 开蛋糕盒 **83%**（200 demos）；DP / IMLE / scratch 均为 **0%** |

- **局限（§7）：** 机器人 DoF 限制高灵巧迁移；缺触觉；预训练规模仍可扩展。

**对 wiki 的映射：** 用对照表写清相对 EgoScale（mid-training 对齐）与 EgoWAM（世界目标置换）的定位。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-egosteer.md`**（流程总览 + 源码运行时序图）。
- 新建 **`sources/sites/egosteer-github-io.md`** 与三仓库归档。
- 更新 **`wiki/methods/egoscale.md`**、**`wiki/entities/paper-egowam-egocentric-human-wam-co-training.md`**、**`wiki/methods/vla.md`**、**`wiki/methods/dagger.md`**、**`wiki/tasks/manipulation.md`**、**`wiki/overview/vla-open-source-repro-landscape-2025.md`** 交叉引用。
- 机构：沿用 `pku` / `upenn`；新增 `psibot`。
