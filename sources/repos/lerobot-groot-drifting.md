# LeRobot · GR00T Drifting（RealManShao fork）

> 来源归档

- **标题：** LeRobot — GR00T N1.7 Drifting Action Head（feat/drif-ov）
- **类型：** repo
- **链接：** <https://github.com/RealManShao/lerobot/tree/feat/drif-ov>
- **上游：** <https://github.com/huggingface/lerobot>
- **权重/数据：** <https://huggingface.co/Xihe666/models>
- **论文：** <https://arxiv.org/abs/2609.18108>
- **入库日期：** 2026-09-17
- **一句话说明：** GR00T N1.7 单步 Drifting action head 实验分支：保留 VLM backbone，替换 flow-matching DiT 为 one-step conditional transformer。
- **代码：** **已开源**（fork 分支 `feat/drif-ov`）
- **沉淀到 wiki：** [`paper-groot-drifting-action-head`](../../wiki/entities/paper-groot-drifting-action-head.md)、[`lerobot`](../../wiki/entities/lerobot.md)
- **交叉归档：** [`groot_drifting_action_head_arxiv_2609_18108.md`](../papers/groot_drifting_action_head_arxiv_2609_18108.md)

## 核心入口

| 路径 | 说明 |
|------|------|
| `src/lerobot/policies/drifting/` | Drifting policy 实现 |
| `src/lerobot/policies/groot/` | GR00T N1.7 基线对照 |
| `docs/source/drifting.mdx` | 安装、训练目标、与 GR00T 对比表 |
| `Experiment-result/LIBERO_latency_stats/` | 延迟与成功率 CSV |

## 复现要点

- Drifting **不加载** GR00T flow-matching action-head 预训练权重；action head 需单独在机器人示范上训练。
- 部署：零 action seed + **一次** conditional transformer 前向 → 完整 chunk；无迭代去噪。
- 训练：proposal（零 seed）+ proximal（`proximal_time` 附近噪声 action）双预测 + geometry-weighted potential loss。
