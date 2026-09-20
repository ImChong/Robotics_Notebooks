# VLA Knows Its Limits（arXiv:2602.21445 · AutoHorizon）

> 来源归档（paper）

- **标题：** VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2602.21445>
- **PDF：** <https://arxiv.org/pdf/2602.21445>
- **项目页：** <https://hatchetproject.github.io/autohorizon/>
- **代码：** <https://github.com/hatchetProject/AutoHorizon>（Apache-2.0）
- **机构：** 伊利诺伊大学芝加哥分校（UIC）；思科研究（Cisco Research）
- **发表：** ECCV 2026（arXiv comment）
- **入库日期：** 2026-09-20
- **一句话说明：** 首个 test-time 方法 **AutoHorizon**：用 flow-based VLA 的 action self-attention 作「预测极限」代理，为每个 action chunk **动态估计 execution horizon**，在 LIBERO / RoboTwin / 真机 manipulation 上接近 oracle 且几乎无额外算力。

## 开源状态

- **已开源（方法实现）：** [hatchetProject/AutoHorizon](https://github.com/hatchetProject/AutoHorizon) 含 **PyTorch π0.5 + LIBERO** 评测栈（基于 [OpenPI](https://github.com/Physical-Intelligence/openpi)）；需自行下载 `gs://openpi-assets/checkpoints/pi05_libero` 并转 PyTorch。
- **权重：** 无 AutoHorizon 专用 HF；沿用 OpenPI 官方 checkpoint。
- **Hugging Face：** 截至入库日 **无官方 HF 页面**（与用户核查一致）。

## Abstract（arXiv）

Action chunking 已是 flow-based VLA 标配，但 **execution horizon**（每段 chunk 实际执行多少步）影响大且研究不足。固定 horizon 下性能先升后降。attention 分析揭示：(i) chunk 内动作对 vision–language token 注意力近乎不变，难适应环境变化；(ii) 首尾 action token 作稳定 anchor，中间动作围绕其组织。据此把 **action self-attention** interpret 为模型预测极限的 proxy，提出 **AutoHorizon**——在 test-time 为每个 chunk 动态估计 execution horizon。仿真与真机 manipulation 上有效、开销可忽略，并可泛化到多样任务与 flow VLA。

## 核心摘录

1. **问题设定：** \(p\) = prediction horizon（chunk 长度），\(e\) = execution horizon（开环执行步数）；\(e/p\) 固定比例 oracle 在不同 \(p\) 下最优比例不同，说明 **静态 horizon 次优**。
2. **AutoHorizon（Elastic）：** 读 action self-attention 分布 → soft-pointer 估计每 chunk 的 \(e\)；稳定段（reach/transport）horizon 拉长，接触/放置等交互段缩短以提高反应性。
3. **基线：** Static Oracle（固定 \(e/p\)）、Static Oracle+（任务调参上界）、Random、Fixed `--replan_steps N`、Action trigger（连续 action delta 阈值）、Uncertainty（多样本 per-step std）。
4. **π0.5 + LIBERO（项目页）：** \(p{=}50\) 时 AutoHorizon **91.6–98.0%** 子集成功率，接近 Static Oracle+；\(p{=}10\) 亦优于或持平固定比例。
5. **RoboTwin + 真机视频：** 多任务上 AutoHorizon 接近或超过 per-task oracle；真机 demo 显示交互阶段 horizon 自适应缩短。

**对 wiki 的映射**

- [paper-autohorizon](../../wiki/entities/paper-autohorizon.md)
- [autohorizon-project.md](../sites/autohorizon-project.md)
- [autohorizon.md](../repos/autohorizon.md)
- [receding-horizon-policy-execution](../../wiki/concepts/receding-horizon-policy-execution.md)
- [action-chunking](../../wiki/methods/action-chunking.md)
