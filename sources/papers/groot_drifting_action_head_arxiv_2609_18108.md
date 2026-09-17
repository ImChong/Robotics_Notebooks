# GR00T Drifting Action Head（arXiv:2609.18108）

> 来源归档（paper）

- **标题：** Technical Report: One-Step Drifting Action Heads for GR00T N1.7
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18108>
- **PDF：** <https://arxiv.org/pdf/2609.18108>
- **代码：** <https://github.com/RealManShao/lerobot/tree/feat/drif-ov>
- **权重/数据：** <https://huggingface.co/Xihe666/models>
- **作者：** Xihe Shao
- **入库日期：** 2026-09-17
- **一句话说明：** GR00T N1.7 迭代 flow-matching DiT action head 换为单步 Drifting head，action head 45.3→5.0 ms，但 LIBERO 三套件成功率系统性下降——速度–成功率 trade-off 审计报告。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-17）：LeRobot fork `feat/drif-ov` 含 `policies/drifting` 实现与文档；HF `Xihe666/models` 托管 checkpoint / 时序统计。

## 核心摘录

1. **动机：** 单步 action 生成可降 VLA 推理成本，但对闭环成功率影响需独立审计。
2. **改动：** 保留 Cosmos-Reason2/Qwen3-VL backbone 与 GR00T 预处理；**替换** flow-matching DiT 为 **one-step Drifting transformer**；**不**复用原 action-head 权重。
3. **延迟（LIBERO 测）：** action head **45.3→5.0 ms**；backbone+head **70.0→30.6 ms**（2×A800 训练环境）。
4. **成功率（3 seeds）：** Spatial **64.0±4.0%**、Goal **52.0±1.0%**、Long **26.0±2.6%** — 低 seed 方差说明非纯初始化噪声。
5. **结论定位：** 作者明确报告为 **speed–success trade-off**，非整体改进；因素含 deterministic one-step mode averaging、batch-dependent geometry、长 open-loop chunk、同步 LIBERO 未测 async overlap。
6. **工程：** 提供 overlap-conditioned 扩展用于异步 chunk 替换；原生 GR00T RTC 在 Drifting 上 **不支持**。

**对 wiki 的映射**

- [paper-groot-drifting-action-head](../../wiki/entities/paper-groot-drifting-action-head.md)
- [lerobot](../../wiki/entities/lerobot.md)
