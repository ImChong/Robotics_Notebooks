# Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation?（arXiv:2609.01453）

> 来源归档（ingest）

- **标题：** Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds
- **短名：** ParcelStow
- **类型：** paper / imitation-learning / temporal-robustness / dexterous-manipulation / unitree-g1
- **arXiv：** <https://arxiv.org/abs/2609.01453>
- **HTML：** <https://arxiv.org/html/2609.01453>
- **PDF：** <https://arxiv.org/pdf/2609.01453>
- **代码：** <https://github.com/coenwerem/parcelstow>（Apache-2.0）— 归档见 [`sources/repos/coenwerem-parcelstow.md`](../repos/coenwerem-parcelstow.md)
- **数据/权重：** <https://huggingface.co/datasets/cenwerem/parcelstow>（Apache-2.0）— 归档见 [`sources/datasets/cenwerem-parcelstow.md`](../datasets/cenwerem-parcelstow.md)
- **作者：** Clinton Enwerem、John S. Baras、Calin Belta
- **机构：** 马里兰大学学院公园分校（University of Maryland）
- **入库日期：** 2026-09-02（公众号浅入库）；**2026-09-03** 按原文 + GitHub/HF 再核
- **索引来源：** [具身智能小站 7 篇盘点](../blogs/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md)
- **一句话说明：** 在 Isaac Lab 的 Unitree G1（腰 + 右臂 + L6 五指，16 关节）ParcelStow 任务上，比较脚本专家与 ACT：标称速度均 100%，示范最高速处专家 84% / ACT 53%。

## 开源状态（步骤 2.5，2026-09-03 再核）

| 资源 | 状态 | 说明 |
|------|------|------|
| GitHub | **已开源（Apache-2.0）** | `coenwerem/parcelstow`：Isaac Lab 扩展、`scripts/run_task.py` / `evaluate.py` / `reproduce.py`、frozen v1 评测记录 |
| 稳定标签 | `v1.0.0` | 对应 arXiv v1 **仅包裹插入**；`main` 另含直立放置与键控插销（未打 v2） |
| Hugging Face | **已发布** | `cenwerem/parcelstow`：Parquet / `.pt` 示范、ACT checkpoint、视频（约 3.94 GB，970,565 行） |
| 真机 | **无** | 全部结果来自 Isaac Lab；数据集卡写明无 sim2real |

**结论：已开源（评测代码 + 示范 + 权重 + CPU 复现脚本）。** v1 论文数字以包裹插入为准；直立 / 插销是 `main` 开发结果，不构成论文主对照。

## 核心摘录（面向 wiki 编译）

### 摘录 1：时间鲁棒性主结论（Abstract / §评测）

- 脚本专家与 ACT 在标称速度 \(r=1\) 均为 **100/100**；示范范围最高速 \(r=2\) 处专家 **84%**、ACT-A **53%**。
- 两个不同初始化 ACT 从标称到最高速分别掉 **34 / 48** 个百分点，专家只掉 **16**。
- \(r=2\) 时 ACT 47 次失败中 **35 次为插入错位**。
- 相对运动交接后，ACT 每次获取都能在空中完成重定向与传递，但只有 **64%** 完成全任务（专家获取后 **95%**）。
- 全部策略与速度下，**414 次无 force-closure 的获取无一完成任务**。

**对 wiki 的映射：** [paper-parcelstow](../../wiki/entities/paper-parcelstow.md)

### 摘录 2：本体、接口与任务冻结（TASK_SPEC / README）

- 平台：**Unitree G1**，骨盆固定；动作是 16 维绝对关节位置（腰 yaw/roll/pitch + 右臂 7 + 拇指 2 + 四指 MCP），50 Hz，物理 200 Hz。
- 观测 147 维：相对关节位/速、上一动作、骨盆系物体位姿、指尖位置与接触力、`task_phase`、`task_rate=r`。
- 包裹：**80×55×40 mm / 0.120 kg**；获取后操作段按时速因子 \(r\) 缩放，获取段固定时长以隔离下游时间需求。
- 示范覆盖 \(r\in[0.5,2.0]\)（297 成功专家回合）；评测网格另含 \(2.25/2.5/3.0\) 做外推。

**对 wiki 的映射：** [Unitree G1](../../wiki/entities/unitree-g1.md)、[Action Chunking](../../wiki/methods/action-chunking.md)

### 摘录 3：三任务开发分支 vs 论文 v1（HF / README，2026-09-03）

| 任务 | Gym ID | 示范 \(r\) | \(r=1\) 专家 / ACT | 额外读法 |
|------|--------|-----------|-------------------|----------|
| 包裹插入（**v1 主文**） | `ParcelStow-L6-Distill-Play-v0` | \([0.5,2.0]\) | 100 / 100（ACT-A） | \(r=2\)：84 / 53 |
| 直立放置（`main`） | `UprightPlace-L6-Play-v0` | \([0.75,1.75]\) | 92 / 39 | \(r=1.75\)：90 / 74；**非**标称匹配对照 |
| 键控插销（`main`） | `PegInsert-L6-Play-v0` | \([0.5,1.0]\) | 93 / 75 | \(r\ge 1.5\) 时 ACT 获取 **0/100** |

包裹插入另有 Diffusion Policy / DAgger checkpoint（`download_artifacts.py --paper`）。直立与插销没有匹配的 DP/DAgger。

**对 wiki 的映射：** [paper-parcelstow](../../wiki/entities/paper-parcelstow.md)、[cenwerem-parcelstow](../datasets/cenwerem-parcelstow.md)

## 当前提炼状态

- [x] 仓库 / HF / TASK_SPEC 已交叉核查（2026-09-03）
- [x] wiki 映射：`wiki/entities/paper-parcelstow.md` 深化
