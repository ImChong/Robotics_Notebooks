# PointCast: One World Model for Rigid, Articulated, and Deformable Object Manipulation

> 来源归档（ingest）

- **标题：** PointCast: One World Model for Rigid, Articulated, and Deformable Object Manipulation
- **类型：** paper / world-model / point-set / diffusion-transformer / manipulation / MPC
- **arXiv：** [2609.28393](https://arxiv.org/abs/2609.28393)
- **项目页：** <https://pointcast-wm.github.io/>
- **venue：** **ICRA 2027 投稿**（项目页 double-anonymous；作者信息 omitted）
- **代码：** **待发布** — 匿名审稿页 **无 GitHub**（2026-09-25）
- **入库日期：** 2026-09-25
- **一句话说明：** **19.8M** 参数 **点集世界模型**：物体+末端执行器持久 3D 点身份轨迹监督；**DiT** 去噪未来点窗口；local/global 交替注意力 + actor cross-attention；**一套架构**覆盖 rigid / cloth / rope / cabinet；仿真 **四 regime 中三项最佳**；真机 PGND **六类中四项 mean 最佳**；冻结 WM **MPC** 四任务 64 episodes 规划。

## 核心摘录

### 状态与预测

- **Mesh-free** 点集状态；每点 **身份持续**，监督 **各自轨迹**（非仅整体形状）。
- 条件：点历史 + **commanded end-effector motion**；输出短窗口未来点位置；rollout 时 append history 再条件化。

### 骨干

- Diffusion Transformer；8 blocks：**kNN-local（16 邻）** 与 **global（register tokens）** 交替；每 block **cross-attend actor tokens**（夹爪上的 magenta actor 点）。

### 评测要点（站页叙事）

- **仿真：** rigid push / cloth lift / rope push / cabinet articulation；Table I 四 baseline 对比 → **3/4 regime 第一**，rigid **第二**。
- **真机 PGND：** cloth / rope / box / paper bag / plush 等；**4/6 category mean 最佳**，另两类第二；优于数据集自带模型 **六类全胜**。
- **Zero-shot：** 仿真 checkpoint 无 real 训练 → **4 captures 中 2 个 best**。
- **MPC：** 每窗口 **一次**网络评估；SE(2) push、顺序 articulation、dragging 等 **64 episodes** 与 baseline 竞争。

## 对 wiki 的映射

- 实体：**[`paper-pointcast-point-set-world-model.md`](../../wiki/entities/paper-pointcast-point-set-world-model.md)**
- 站点：**[`pointcast-wm-github-io.md`](../sites/pointcast-wm-github-io.md)**
- 交叉：[`generative-world-models`](../../wiki/methods/generative-world-models.md)、[`model-based-rl`](../../wiki/methods/model-based-rl.md)、[`manipulation`](../../wiki/tasks/manipulation.md)
