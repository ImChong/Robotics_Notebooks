# Walk Like This: Stylized Locomotion for Physics-Based Characters through Example-Driven Diffusion

> 来源归档（ingest · arXiv 论文 + Disney Research 项目页）

- **类型：** paper / humanoid-locomotion / motion-generation / diffusion / sim2real
- **会议：** CoRL 2026
- **arXiv：** <https://arxiv.org/abs/2610.18147>
- **项目页：** <https://la.disneyresearch.com/publication/walk-like-this-stylized-locomotion-for-physics-based-characters-through-example-driven-diffusion/>
- **作者：** Joshua Aurand、Agon Serifi、Sammy Christen、Ruben Grandia、Espen Knoop、Marco Hutter、Moritz Bächer
- **机构：** Disney Research Zürich；ETH Zürich
- **入库日期：** 2026-10-20
- **一句话说明：** 从无标签动作数据学习示例风格编码器与自回归扩散运动生成器，再由 RL tracker 在 Unitree G1 上实时跟踪；用户可独立控制速度和动作风格。

## 核心论文摘录

### 1) 风格与功能运动解耦（Aurand et al., 2026）
- **链接：** <https://arxiv.org/abs/2610.18147>
- **核心贡献：** 联合训练 Transformer 风格编码器与自回归扩散模型；训练时从同一动作序列独立采样风格片段和待预测片段，避免编码器把具体动作内容当成风格。推理时可用不同长度、训练未见过的动作示例作为风格条件。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-walk-like-this-stylized-locomotion.md)；[扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)

### 2) 推理期可控性与速度跟随
- **链接：** <https://arxiv.org/abs/2610.18147>
- **核心贡献：** 在注意力 logits 中提高风格 token、降低历史运动 token 的权重，可在推理期调节风格强度并快速切换；反馈修正 joystick 速度指令，减小自回归生成造成的轨迹漂移。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-walk-like-this-stylized-locomotion.md)；[类人机器人运动](../../wiki/tasks/humanoid-locomotion.md)

### 3) 生成运动到实体机器人
- **链接：** <https://arxiv.org/abs/2610.18147>
- **核心贡献：** 用扩散模型自回归 rollout 生成的数据训练 PPO 跟踪策略，并在 Unitree G1 上零样本部署。论文报告约 4 小时硬件测试无跌倒；Jetson Thor 上扩散推理约 11 ms/步、指令到动作约 17 ms。
- **对 wiki 的映射：** [论文实体](../../wiki/entities/paper-walk-like-this-stylized-locomotion.md)；[类人机器人运动](../../wiki/tasks/humanoid-locomotion.md)

## 开源状态核查（2026-10-20）

打开 Disney Research 项目页及 arXiv 页面：项目页提供论文、摘要和补充视频，未列 GitHub、数据集或模型权重链接；arXiv 提供论文和 TeX 源码。**可公开阅读论文与视频，但未发现可运行的训练 / 推理代码或数据发布入口**。因此 wiki 源码运行时序图标为不适用。
