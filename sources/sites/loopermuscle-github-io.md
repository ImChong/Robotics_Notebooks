# loopermuscle.github.io（LooperMuscle 项目页）

- **标题：** LooperMuscle: Fast and Stable Learning of Humanoid Whole-Body Tracking via Structured Mixture-of-Experts
- **类型：** site / project-page
- **URL：** <https://loopermuscle.github.io/>
- **配套论文：** [LooperMuscle（arXiv:2608.00820）](https://arxiv.org/abs/2608.00820) — 归档见 [`sources/papers/loopermuscle_arxiv_2608_00820.md`](../papers/loopermuscle_arxiv_2608_00820.md)
- **代码：** <https://github.com/LooperMuscle/Code> — 归档见 [`sources/repos/loopermuscle-code.md`](../repos/loopermuscle-code.md)
- **机构：** 深镜智能（DeepMirror Inc.）；香港科技大学（HKUST）；穆罕默德·本·扎耶德人工智能大学（MBZUAI）
- **入库日期：** 2026-08-22

## 一句话摘要

LooperMuscle 官方项目页：展示 **FastSAC 墙钟加速脉络下结构化 MoE 全身跟踪** 的训练曲线与 G1 真机格斗序列演示；Code 按钮指向 GitHub 仓。

## 公开信息要点（截至入库日）

- **作者：** Boyi Liu、Qijin Li、Tianqi Yu、Qinrui Yan、Xingxing Zuo（通讯）。
- **摘要要点：** 语义结构化 MoE actor、专家感知分布式 critic、贡献路由 replay + 延迟课程；~45 min 收敛，相对 FastSAC 奖励 +47.5%，相对 PPO 约 8× 墙钟加速。
- **演示：** 训练 reward 曲线；G1 真机 KungfuBot2 动作库全身跟踪。

## 开源核查（步骤 2.5）

| 项 | 状态（2026-08-22） |
|----|-------------------|
| 项目页 Code 链 | 指向 [LooperMuscle/Code](https://github.com/LooperMuscle/Code) |
| 已发布 | Holosoma 推理/真机 WBT、运动重定向、Holosoma 训练框架（PPO/FastSAC） |
| 论文 MJLab 基准 | 特权观测接口；与真机 Holosoma 154-D 接口不同，需重训 |
| 结论 | **部分开源**（部署与 Holosoma 训练可跑；MJLab 论文数字与可部署策略需分读） |

## 为何值得保留

- **非 PDF 证据：** 真机格斗序列与训练曲线是速度–质量权衡的直观判据。
- **与 GitHub 三角互证：** 项目页 Code 链与仓 README 一致。
- **FastSAC 生态锚点：** 把 off-policy 墙钟加速从 locomotion 延伸到 WBT 的代表工作。

## 关联资料

- 论文归档：[`sources/papers/loopermuscle_arxiv_2608_00820.md`](../papers/loopermuscle_arxiv_2608_00820.md)
- 代码仓库：[`sources/repos/loopermuscle-code.md`](../repos/loopermuscle-code.md)
