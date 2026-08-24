# NaP-Control: Navigating Diffusion Prior for Versatile and Fast Character Control

> 来源归档（ingest）

- **标题：** NaP-Control: Navigating Diffusion Prior for Versatile and Fast Character Control
- **类型：** paper
- **机构：** ETH Zurich
- **论文链接：** <https://arxiv.org/abs/2605.20209>
- **项目页：** <https://chiawenchen.github.io/nap-control-project/>
- **代码：** <https://github.com/chiawenchen/NaP>
- **入库日期：** 2026-08-24
- **一句话说明：** 用 PPO 在冻结扩散动作先验的初始噪声空间上导航，替代测试时梯度引导，实现快速、多任务的物理角色全身控制。

## 核心摘录（策展，非全文）

1. **问题：** 扩散策略（UniPhys、DiffuseCLoC 等）表达力强，但任务适配常依赖可微目标 + 测试时迭代梯度引导，推理慢且脆弱；纯离线训练难以应对闭环接触与失稳。
2. **方法：** 预训练任务无关的因果 Transformer 扩散动作先验（AMASS 跟踪数据，沿用 UniPhys 范式 + 更紧凑状态表示）；冻结先验后，用 PPO 学习 **初始 latent noise** 导航策略（DSRL 思路扩展到高维全身物理控制）；DDIM 5 步去噪 + PULSE latent action decoder → PD 力矩；action chunk 开环执行（k=4/8）。
3. **任务：** 远目标到达、敏捷右手到达、速度跟踪、沙发坐姿交互；可选地形高度图泛化到未见崎岖地形（课程学习）。
4. **结果（论文）：** 相对 UniPhys 远目标推理 **22.5 vs 2.9 FPS**、成功率 **98.4% vs 81.9%**；jerk 与任务成功率在平坦/崎岖多任务上优于 PULSE、CLoSD、MaskedMimic 等；远目标平坦场景优于 AMP/CML 等任务专用对抗先验。
5. **开源（2026-08-24 核查）：** 官方仓库含 Isaac Gym 环境、先验/导航策略训测脚本、`download_data.sh` / checkpoint 下载与 `nap/evaluation/run_evaluate.py`；依赖 SMPL 模型与 Isaac Gym Preview 4（用户自备）。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | `wiki/entities/paper-nap-control.md` |
| 扩散先验对照基线 | `wiki/entities/paper-bfm-40-uniphys.md` |
| 角色动画 × 机器人边界 | `wiki/concepts/character-animation-vs-robotics.md` |
| AMP / motion prior 谱系 | `wiki/overview/humanoid-amp-motion-prior-survey.md` |

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2605.20209>
- 项目页：<https://chiawenchen.github.io/nap-control-project/>
- 代码：<https://github.com/chiawenchen/NaP>
