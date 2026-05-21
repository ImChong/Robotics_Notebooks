# interprior_arxiv_2602_06035

> 来源归档（ingest）

- **标题：** InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions
- **类型：** paper
- **来源：** arXiv:2602.06035（HTML v1；项目页标注 CVPR 2026 Highlight）
- **入库日期：** 2026-05-17
- **一句话说明：** 面向 **物理仿真中人–物全身交互（HOI）** 的 **生成式运动先验**：先用 **全参考模仿专家（InterMimic+，PPO）** 吸收大规模数据，再 **蒸馏** 为 **稀疏目标条件变分策略**，最后用 **带正则与失败态重置的 RL 微调** 把「数据内重建」巩固为可泛化流形，支持未见目标/初始化、未见物体上的组合技能与用户交互控制，并在 **Unitree G1** 上做 sim-to-sim 与键盘实时控制演示。

## 核心论文摘录（MVP）

### 1) 问题动机与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2602.06035> · PDF：<https://arxiv.org/pdf/2602.06035>
- **核心贡献：** 人–物交互是 **分层** 的：高层稀疏意图（ affordance / 接触目标）应能诱导 **平衡、接触与操作** 的协调涌现，而非依赖稠密全身参考。InterPrior 用 **大规模模仿预训练 + RL 后训练** 学习统一 **生成式控制器**；指出 **纯蒸馏** 在 HOI **组合构型空间** 上易脆，**孤立 RL** 又易 unnatural reward hacking，故采用 **强模仿初始化 + 作为局部优化器的 RL** 二者互补。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [InterPrior 论文实体](../../wiki/entities/paper-interprior.md)

### 2) 三阶段总览与 MDP 接口（Sec.3 / Fig.2）

- **链接：** <https://arxiv.org/html/2602.06035v1#S3>
- **核心贡献：** **(I)** 训练全参考模仿专家 \(\pi_E\)（大尺度 HOI 数据 + 数据增广与物理扰动 + 塑形奖励）；**(II)** 将专家蒸馏为 **掩码条件变分策略** \(\pi\)，从 **多模态观测 + 高层意图** 重建运动；**(III)** **RL 微调** 提升未见目标与初始化的成功率，并用 **正则** 锚定在预训练模型上。三阶段共享一致的 **观测 / 目标条件 / 低层驱动动作** MDP 表述。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)

### 3) InterMimic+：全参考模仿专家（Sec.3.2）

- **链接：** <https://arxiv.org/html/2602.06035v1#S3.SS2>
- **核心贡献：** 在 **InterMimic** 式人–物 **共跟踪** 上增强：奖励基线为 \(r = r_{\text{track}} \times r_{\text{energy}}\)；通过 **人–物初始位姿随机化、稀疏冲量扰动、物体形状与惯量/摩擦等物理随机化** 扩大参考覆盖；引入 **无参考手奖励** \(r_h\) 使手围绕 **仿真中真实物体几何** 对齐/包络，减轻「薄小物体」上盲跟参考的精度塌缩；**终止惩罚** \(r_{\text{ter}}\) 抑制跌倒与大偏差下的投机行为。
- **对 wiki 的映射：**
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)（专家阶段等价于「全参考特权监督」叙事）

### 4) 变分蒸馏：掩码多模态目标 + 潜变量几何约束（Sec.3.3）

- **链接：** <https://arxiv.org/html/2602.06035v1#S3.SS3>
- **核心贡献：** 观测含人体/物体运动学与 **段–面有符号距离、接触指示** 等交互项（根坐标 + 局部航向归一化）。目标为 **短_horizon 预览序列 + 长_horizon 快照** 的 **掩码残差编码**（旋转用 log-map）。策略为 **prior / encoder–decoder VAE 结构**：训练期 encoder 读完整未来参考，与 **历史 + 稀疏目标** 的 prior 组合成 **残差后验**；推理期仅从 prior 采样潜变量。**episode 内固定噪声** 以促进时间连贯；采样后对 \(z_t\) 做 **球面投影** 抑制 OOD 潜变量；辅以 **在线蒸馏（DAgger 式）** 与 **KL / 目标重建辅助头**。
- **对 wiki 的映射：**
  - [DAgger](../../wiki/methods/dagger.md)
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)（与扩散式先验并列的「潜变量生成控制」参照）

### 5) RL 微调：未见条件、失败态合成与知识保持（Sec.3.4）

- **链接：** <https://arxiv.org/html/2602.06035v1#S3.SS4>
- **核心贡献：** 在 **未见目标与初始化** 上优化成功相关回报，并以 **正则** 限制偏离预训练行为；利用 **预训练策略生成自然过渡**，从 **失败状态重置** 采样以学习 **再接近 / 再抓取** 等恢复行为，将离散技能 **嵌入连续可泛化流形**；论文报告可推广到 **未见物体** 的交互，并展示 **中途切换命令、长时域随机切换目标** 等。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)（文中主证据为仿真内泛化 + G1 sim-to-sim / 交互控制演示，真机部署为「潜力」叙述）

## 其他公开资料（非 PDF 正文）

- **项目页（视频、能力条、BibTeX）：** <https://sirui-xu.github.io/InterPrior/> — 归档见 [sources/sites/sirui-xu-interprior-github-io.md](../sites/sirui-xu-interprior-github-io.md)
- **同系列基础工作 InterMimic（CVPR 2025 Highlight）：** <https://sirui-xu.github.io/InterMimic/>
- **InterAct 数据集（大规模 HOI 基准）：** <https://sirui-xu.github.io/InterAct/>
- **ULTRA（多模态人形 loco-manipulation，站点链到「Deploy InterPrior to real」叙事）：** <https://ultra-humanoid.github.io/>

## 当前提炼状态

- [x] 论文摘要与核心方法摘录
- [x] wiki 页面映射与姊妹资料互链
