# GameScaling：多样虚拟世界中的具身智能体持续预训练（arXiv:2610.28698）

> 来源归档（ingest）

- **标题：** Scaling in Games: Continued Pre-Training for Embodied Agents in Diverse Virtual Worlds
- **简称：** GameScaling
- **类型：** paper / game agents / embodied agents / continued pre-training
- **arXiv：** [2610.28698](https://arxiv.org/abs/2610.28698)（v1，2026-10-20）
- **作者：** Kuan Zhang、Yukun Chen、Zhihao Yang、Yue Su、Xiangnan Wu、Zirong Chen、Run Luo、Jinkun Hou、Tao Tan、Yinhe Zheng、Yiming Li
- **机构：** miHoYo Honkai AI R&D Team、清华大学、中国科学院大学、香港大学、新加坡国立大学、北京大学
- **项目页：** https://gamescaling.github.io/
- **代码：** https://github.com/THUSI-Lab/GameScaling（推理客户端；无训练代码）
- **模型：** https://huggingface.co/collections/THUSI-Lab/gamescaling-model
- **入库日期：** 2026-10-23
- **一句话说明：** 受控研究游戏数据量、世界多样性和模型规模对具身游戏智能体持续预训练及跨世界泛化的影响。

## 摘要级要点

- **研究设计：** 120 组训练配置，覆盖 0.8B–27B 四种模型规模、50–1,600 小时六档数据预算，以及从 Minecraft 单世界到多游戏混合的五种数据组合；训练语料涉及 80 款游戏。
- **评测范围：** Minecraft 与 ViZDoom 为训练内世界；OmniGameArena 是动作相近的未见 3D 世界；GameWorld 代表视觉与控制方式差异更大的远域世界。
- **主要发现：** 更多数据改善语料内表现；多样性更有助于迁移到相近世界，对远域迁移帮助有限；单世界训练下扩大模型会带来远域泛化劣势。
- **训练阶段观察：** 早期更像学习键鼠接口，带来较广泛的跨世界收益；后期收益逐渐集中到训练内及相近世界。
- **局限：** 持续预训练模型在 3D 世界具竞争力，但远域差距仍在；作者指出仅靠持续预训练不足以得到通用具身智能体。

## 核心论文摘录（MVP）

### 1) 数据、多样性与模型规模的受控比较

- **链接：** [论文](https://arxiv.org/abs/2610.28698)
- **摘录要点：** 研究沿数据量、游戏混合多样性与模型参数规模三条轴比较 120 组配置，而非只比较单一模型与基线。

### 2) 由近及远的跨世界评测

- **链接：** [项目页](https://gamescaling.github.io/)
- **摘录要点：** 评测从训练语料内游戏扩展到动作相近的未见 3D 世界，再到视觉和控制范式更远的 2D/2.5D 游戏；迁移收益随世界差异增大而减弱。

### 3) 实时交互中的延迟效应

- **链接：** [项目页演示](https://gamescaling.github.io/)
- **摘录要点：** 项目页区分暂停决策与实时决策；实时游戏不会等待模型推理，因此延迟也影响分数。页面报告 GameScaling 每步约 0.13–0.69 秒，所比较前沿模型约 4–24 秒，具体口径以论文评测协议为准。

## 开源状态

截至 2026-10-23，官方公开推理客户端和八个模型检查点，代码与权重标注 Apache-2.0；训练语料不公开，仓库也明确没有训练代码。Apache-2.0 不覆盖游戏内容、画面、版权或商标。

## 对 wiki 的映射

- 实体页：[GameScaling](../../wiki/entities/paper-gamescaling.md)
- 同领域基准：[GameHorizon Suite](../../wiki/entities/paper-gamehorizon-suite.md)
- 相关主题：游戏智能体、持续预训练、跨世界泛化、实时交互延迟
