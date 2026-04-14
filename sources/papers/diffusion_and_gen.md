# diffusion_and_gen

> 来源归档（ingest）

- **标题：** Diffusion Policy 与生成式模仿学习核心论文
- **类型：** paper
- **来源：** RSS / CoRL / arXiv / Science Robotics
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 Diffusion Policy、π₀、BESO 等扩散模型在机器人模仿学习中的应用

## 核心论文摘录

### 1) Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（Chi et al., RSS 2023）
- **链接：** <https://arxiv.org/abs/2303.04137>
- **核心贡献：** 把 DDPM 扩散过程用于动作生成；输入：视觉观测 + 当前状态；输出：未来动作序列（action chunk）；优于 BC/IBC/LSTM；处理多模态分布（同一状态多种合理动作）
- **关键公式：** $a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(a_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(a_t,t,o)\right) + \sigma_t z$
- **对 wiki 的映射：**
  - [diffusion-policy](../../wiki/methods/diffusion-policy.md)
  - [imitation-learning](../../wiki/methods/imitation-learning.md)

### 2) π₀: A Vision-Language-Action Flow Model for General Robot Control（Black et al., 2024）
- **链接：** <https://arxiv.org/abs/2410.24164>
- **核心贡献：** 基于 flow matching（连续时间扩散变体）训练通用机器人基础模型；结合 VLM 做高层任务理解；在多任务、多机器人形态上实现 zero-shot 迁移；Physical Intelligence 代表工作
- **关键洞见：** Flow matching > DDPM（更快采样，训练更稳定）；VLM 做条件化解决任务泛化
- **对 wiki 的映射：**
  - [diffusion-policy](../../wiki/methods/diffusion-policy.md)
  - [imitation-learning](../../wiki/methods/imitation-learning.md)
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)

### 3) BESO: Bench-top Robot Learning via Score-based Diffusion Policies（Reuss et al., 2023）
- **链接：** <https://arxiv.org/abs/2304.02532>
- **核心贡献：** 使用 score-based diffusion（能量梯度形式）；引入 CLIP 语言条件化支持指令跟随；在 Calvin benchmark 上 state-of-the-art；展示扩散策略与语言理解的结合路径
- **对 wiki 的映射：**
  - [diffusion-policy](../../wiki/methods/diffusion-policy.md)
  - [imitation-learning](../../wiki/methods/imitation-learning.md)

### 4) ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（Zhao et al., RSS 2023）
- **链接：** <https://arxiv.org/abs/2304.13705>
- **核心贡献：** Action Chunking with Transformers：用 CVAE 生成动作序列块（chunk）；低成本遥操作数据采集（ALOHA）；在精细双臂操作任务上效果显著
- **关键洞见：** Temporal ensemble（多次推理结果平均）减少执行抖动；chunk 大小是关键超参数
- **对 wiki 的映射：**
  - [imitation-learning](../../wiki/methods/imitation-learning.md)
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)

### 5) Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation（Prasad et al., 2024）
- **链接：** <https://arxiv.org/abs/2405.07503>
- **核心贡献：** 将 Consistency Model 蒸馏技术应用于 Diffusion Policy；1 步推理（vs 扩散的 10-100 步）；保持扩散策略的多模态表达能力；推理速度提升 10x，适合实时控制
- **对 wiki 的映射：**
  - [diffusion-policy](../../wiki/methods/diffusion-policy.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
