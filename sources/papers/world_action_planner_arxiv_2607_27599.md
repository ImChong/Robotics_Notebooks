# World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models（arXiv:2607.27599）

> 来源归档（ingest）

- **标题：** World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models
- **缩写：** **WAP** / World Action Planner
- **类型：** paper / action-conditioned world model + VLM planning
- **arXiv：** <https://arxiv.org/abs/2607.27599>
- **PDF：** <https://arxiv.org/pdf/2607.27599>
- **项目页：** <https://worldactionplanner.github.io/>
- **代码：** <https://github.com/XiangchengZhang/world-action-planner>
- **权重/数据：** <https://huggingface.co/XiangchengZhang/world-action-planner>
- **发表日期：** 2026-07-30（arXiv preprint）
- **作者：** Xiangcheng Zhang, Yilun Du
- **机构：** 哈佛大学（Harvard University）
- **入库日期：** 2026-08-02
- **一句话说明：** 用 pose-image 条件多视角世界模型 + VLM 提议/优化/搜索，在想象 rollout 上做模型基规划；组合任务、新布局与零样本场景显著优于 π₀.₅ / cosmos-policy 等 E2E VLA/WAM。

## 核心论文摘录（MVP）

### 1) 问题：E2E 模仿在组合/新布局上失配（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.27599>
- **核心贡献：** 仅靠示教分布的 E2E VLA/WAM 在长程组合、目标物重布局与未见任务上易卡死或回退到训练坐标。World Action Planner 改走 **VLM 提原语 → 动作条件世界模型想象 → 全局反馈优化 + 局部网格搜索**，把策略当可选工具而非唯一决策器。
- **对 wiki 的映射：**
  - [World Action Planner 论文实体](../../wiki/entities/paper-world-action-planner.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [生成式世界模型](../../wiki/methods/generative-world-models.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) Pose-image 条件多视角世界模型（§3.1 / §5.1）

- **核心贡献：**
  - 由动作经正运动学得到未来关节位姿，渲染为 **pose skeleton 图像**（相对低维动作 AdaLN/cross-attn 更利于 OOD 动作）。
  - 多相机拼成 2×2；Wan-T2V-1.3B + **diffusion forcing / flow matching**；pose token 不加噪。
  - 相对 WPE / Ctrl-World 等基线：单具身 in-distribution 平均约 **+11.4%**，泛化设置约 **+16.8%**（PSNR/LPIPS 相对提升，Table 1）。
- **对 wiki 的映射：**
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md)
  - [Wan](../../wiki/entities/paper-wan-video.md)
  - [LIBERO](../../wiki/entities/libero-benchmark.md)

### 3) 规划管线：提议 → 全局优化 → 局部搜索（§3.2 / Alg.1）

- **核心贡献：**
  - VLM（默认 Gemini 3.0 Flash）提 MOVE/ROTATE/GRASP/RELEASE；多视角像素三角化得 3D 目标，低层控制器出动作块。
  - **全局优化：** 世界模型想象轨迹 → VLM 语义反馈（抬高避撞等）修正目标。
  - **局部搜索：** 网格候选 + 想象视频排序；近目标时可再 roll 扩散策略并想象择优。
  - **Policies as tools：** DP / VLA / WAM 仅作可选工具，OOD 走完整规划。
- **对 wiki 的映射：**
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 4) 评测：组合 / 新布局 / 零样本（§5.2）

- **核心贡献：**
  - **组合（Table 3，LIBERO-Long，50 trials）：** WAP 72/68/78/70 vs π₀.₅≈0–4、cosmos-policy≈0、纯 VLM planner 28–56。
  - **新布局（Table 4，LIBERO-Object 改布局）：** WAP 66–90；π₀.₅ / cosmos-policy 多为 0；策略仅用每任务 **5** 条示教。
  - **零样本（Table 5，Robosuite）：** PickPlaceCan **80** / StackCube **76**（无专用示教策略；WM 用 50 条探索轨迹微调）。
  - 理论（§4）：多任务 tabular / 线性 MDP 下，模型基规划相对模仿在任务数随数据预算增长时更可泛化。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [具身 FM 选型闭环](../../wiki/queries/embodied-fm-taxonomy-loop.md)

## 开源状态（2026-08-02 项目页核查）

- **已开源：** 项目页列 GitHub → [XiangchengZhang/world-action-planner](https://github.com/XiangchengZhang/world-action-planner)；权重/数据 → [Hugging Face](https://huggingface.co/XiangchengZhang/world-action-planner)（含 `world_models/*`、`diffusion_policy/*`）。
- 归档：[`sources/sites/worldactionplanner-github-io.md`](../sites/worldactionplanner-github-io.md)、[`sources/repos/world-action-planner.md`](../repos/world-action-planner.md)、[`sources/sites/huggingface-xiangchengzhang-world-action-planner.md`](../sites/huggingface-xiangchengzhang-world-action-planner.md)。

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-world-action-planner.md`](../../wiki/entities/paper-world-action-planner.md)
- 交叉升级：
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [生成式世界模型](../../wiki/methods/generative-world-models.md)
  - [VLA](../../wiki/methods/vla.md)
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [LIBERO](../../wiki/entities/libero-benchmark.md)
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

## 推荐继续阅读

- 项目页 Demo / Algorithm 图：<https://worldactionplanner.github.io/>
- Large Video Planner / Diffusion Forcing Transformer（README 声明架构基础）
