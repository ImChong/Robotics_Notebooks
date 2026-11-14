# WorldScape-Policy（manifoldai-research/WorldScape-Policy）

> 来源归档

- **标题：** WorldScape Policy 2.0 官方仓库
- **类型：** repo（**占位仓库，代码未发布**）
- **来源：** Manifold AI · WorldScape Team（联合清华大学、上海交通大学）
- **链接：** <https://github.com/manifoldai-research/WorldScape-Policy>
- **权重：** <https://huggingface.co/manifoldai-research/WorldScape-Policy-2>（同为占位卡片）
- **论文：** <https://arxiv.org/abs/2607.18840> — 归档见 [`sources/papers/worldscape_policy_2_arxiv_2607_18840.md`](../papers/worldscape_policy_2_arxiv_2607_18840.md)
- **项目页：** <https://manifoldai-research.github.io/WorldScape-Policy/> — 归档见 [`sources/sites/manifoldai-research-worldscape-policy.md`](../sites/manifoldai-research-worldscape-policy.md)
- **许可：** HF 模型卡声明 **Apache-2.0**；GitHub 仓库未附 LICENSE 文件
- **入库日期：** 2026-08-03
- **一句话说明：** 仓库描述为「Empower steerable world action modeling with reasoning-augmented memory and event-grounded pretraining under multimodal control」，但截至入库日只有 README 与 `.gitignore`。
- **沉淀到 wiki：** [`wiki/entities/paper-worldscape-policy-2.md`](../../wiki/entities/paper-worldscape-policy-2.md)

---

## 开放程度核查（2026-08-03）

| 项 | 状态 | 依据 |
|----|------|------|
| 训练代码 | **未发布** | README：*"Code is coming soon. We are preparing the training, inference, and evaluation code for release."* |
| 推理 / 评测代码 | **未发布** | 同上 |
| 预训练权重 | **未发布** | HF 卡片：*"Model is coming soon. We are preparing the pre-training model and post-training checkpoint of RoboTwin 2.0 dataset for release."* |
| RoboTwin 2.0 后训练 checkpoint | **未发布** | 同上 |
| ManipEvent-5M 数据集 | **未见发布计划** | 论文与项目页均未给出数据下载入口 |
| 仓库文件 | `README.md`、`.gitignore` | GitHub 目录树 |

**结论：宣称将开源 / 待发布。** 论文实体页不得写「已开源」；`## 源码运行时序图` 按 **不适用** 处理，待代码发布后再补并与实际入口对齐。

---

## README 声明的能力（供发布后核对）

| 能力 | README 措辞 |
|------|-------------|
| 长程任务规划 | 从高层指令做 long-horizon task planning |
| 事件级子任务跟随 | event-level subtask following |
| 记忆依赖视觉推理 | memory-dependent visual reasoning（含 shell game 类跟踪） |
| 上下文技能迁移 | 目标图 / 演示视频驱动的 in-context skill transfer |
| 统一 video-action 建模 | 在 ManipEvent-5M 上训练 |

三层记忆结构（README 与论文一致）：**短期视觉记忆**（近期观测作 causal prefill）、**长期事件记忆**（global-history / local-active / event-boundary 三视图）、**隐式子目标推理**（检索历史增强感知与规划 token）。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-worldscape-policy-2](../../wiki/entities/paper-worldscape-policy-2.md) | 论文实体与结论 |
| [paper-worldscape-moe-heterogeneous-action](../../wiki/entities/paper-worldscape-moe-heterogeneous-action.md) | 同 Manifold AI；上游异构动作可控视频 WM |
| [robotwin](../../wiki/entities/robotwin.md) | 主评测基准（50 任务 / C2R 协议） |
| [agibot-world-2026](../../wiki/entities/agibot-world-2026.md) | ManipEvent-5M 最大公开真机来源（285.56M 帧） |

## 固定版本复核（2026-11-13）

- **版本：** [148ebfb2a84184eeac2c9ccf36d5b4f9f90e6aa8](https://github.com/manifoldai-research/WorldScape-Policy/tree/148ebfb2a84184eeac2c9ccf36d5b4f9f90e6aa8)
- **核对范围：** 根README、cli/evaluate.py、rollout/runner.py与session.py、evals/agilex/server.py全文；WorldScapePolicy.condition/sample、AutoConditioner.forward/promote_pending完整函数；checkpoint transform检查入口。
- **通读：** 已完成
- 代码/预训和RoboTwin后训检查点、训练recipes/评测/真机工具已发布，旧占位结论不再适用。完整ManipEvent自采数据未见一并开放。
- CLI加载native checkpoint与transform，模式匹配。PolicyRuntime.predict暂存动作及两类记忆，执行成功commit后才推进；异常discard不推进记忆。Policy.sample路由Auto/T5、视觉prefill后交WAM插件采样。Auto先VLM特征与规划token，再历史检索门控。
- README RoboTwin当前评测使用14D绝对关节动作、24/48 horizon、9帧历史，与论文双臂20D EEF原理描述分属不同本体adapter，不能无条件套同一动作归一化。HDF5 replay与manifold实时执行分开，未调用物理硬件。
- 代码Apache2.0，模型/数据依赖按独立许可；静态核读未训练或复现论文成绩。
