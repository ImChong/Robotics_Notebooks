# 前沿 | 人形控制算法 Mimic 整体演进：DeepMimic → AMP → ASE → CALM → PHC → MaskedMimic → BeyondMimic

> 来源归档（blog / 微信公众号）

- **标题：** 前沿 | 人形控制算法 Mimic 整体演进：DeepMimic → AMP → ASE → CALM → PHC → MaskedMimic → BeyondMimic
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/SEBbND3AJiPU-8hrw2aiBw
- **发表日期：** 2026-09-15（入库日；页面未稳定暴露 `publish_time`）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **一句话说明：** 万字综述把物理角色/人形 mimic 主线从「显式 tracking 奖励」演进到「对抗先验 → 技能隐空间 → 条件化 latent → 大规模 tracking/recovery → 掩码补全 → 扩散组合」；强调 **能力演进图** 而非严格软件继承链（PHC 与 CALM 并行、BeyondMimic 重回高保真 tracking + 生成式组合）。

## 核心摘录（归纳，非全文）

### 四个长期矛盾

| 矛盾 | 后续方法的应对方向 |
|------|-------------------|
| 精确跟踪 vs 鲁棒性 | 相对/锚点跟踪、失败恢复、部分约束 |
| 动作规模 vs 人工 reward 成本 | AMP 对抗先验、ASE/CALM 隐空间、MaskedMimic 统一接口 |
| 多样性 vs 可控性 | CALM 条件 latent、BeyondMimic classifier guidance |
| 训练任务 vs 未见任务组合 | ASE 技能嵌入、MaskedMimic 补全、BeyondMimic 扩散蒸馏 |

### 能力演进表（文内 Table 1 摘要）

| 年份 | 方法 | 核心机制 | 代表性能力 |
|------|------|----------|------------|
| 2018 | DeepMimic | 显式 imitation reward + PPO + RSI | 单/少量动作高保真模仿 |
| 2021 | AMP | Adversarial Motion Prior | 无结构动作数据的风格约束 |
| 2022 | ASE | AMP + Mutual Information latent | 可复用连续技能隐空间 |
| 2023 | CALM | Motion Encoder + Conditional Discriminator | 由示范动作直接指定 latent |
| 2023 | PHC | PMCP + AMP + Recovery | 10k 级 tracking、跌倒恢复 |
| 2024 | MaskedMimic | Masked Motion Inpainting + C-VAE | 关节/关键帧/文本/场景部分约束 |
| 2026 | BeyondMimic | RL Tracking + VAE + Latent Diffusion + Guidance | 真机敏捷 tracking、测试时任务组合 |

### 文内关键机制（一页记忆）

- **DeepMimic：** 参考动作是优化目标而非 playback；RSI 解决长时序探索；PD 目标接口被后人形 RL 广泛继承。
- **AMP：** task reward 管 what，motion prior 管 how；与 GAIL 区别在于 prior 来自动捕分布而非逐帧对齐。
- **ASE：** 大量自然动作压缩成可复用技能 embedding；互信息约束 latent 与动作语义。
- **CALM：** 解决 ASE latent 难以精确指定具体动作；encoder 把示范映射到可组合 latent。
- **PHC（并行分支）：** 大规模 motion tracking + 永久恢复；非 CALM 直接后继。
- **MaskedMimic：** 统一控制模态为「部分动作约束下的运动补全」。
- **BeyondMimic：** 高质量真机 tracking 教师 + 状态–动作扩散；测试时 guidance 做零样本组合。

## 对 wiki 的映射

- **新建：** [mimic 控制演进技术地图](../../wiki/overview/mimic-control-evolution-lineage.md)
- **交叉补强：** [DeepMimic](../../wiki/methods/deepmimic.md)、[AMP 运动先验综述](../../wiki/overview/humanoid-amp-motion-prior-survey.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[人形运动小脑技术地图](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md)
