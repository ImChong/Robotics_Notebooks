# Light REACT 项目发布（亮源新创）

- **标题：** Light REACT — Resilient Humanoid Control（全身韧性智能）
- **类型：** site（官方 Tech Blog 发布）
- **机构：** 亮源新创（Light Origins）
- **Tech Blog：** <https://www.lightorigins.com/en/blog/light-react>
- **主发布：** [微信公众号](https://mp.weixin.qq.com/s/Xfps8-XAv3u1S--EpS5Ezw)（2026-09-09）
- **机构主页：** <https://www.lightorigins.com/>
- **入库日期：** 2026-09-21
- **核对范围：** 通读官方 Tech Blog 全文（`https://www.lightorigins.com/en/blog/light-react`，含 01 Scalable Deployment & Resilience Pyramid、02 Three-Stage Training Recipe、03 6 Specialist Teachers across 196 damage cases、04 Multi-Teacher Distillation 架构对比、05 Preference Alignment RL 及真机临时故障/限高恢复演示）；截至 2026-11-21 无 arXiv 论文或公开 GitHub 代码，未进行本地训练复现。
- **通读：** 已完成

## 开源核查（入库日 / 2026-11-21 复核）

| 资产 | 状态 |
|------|------|
| Tech Blog | **已上线** <https://www.lightorigins.com/en/blog/light-react> |
| 论文 / arXiv | **未列** |
| 代码 / 权重 | **未开源**（无 GitHub / HF 链接） |
| 演示视频 | Tech Blog 与微信公众号正文内嵌（未单独归档） |

## 能力摘要

- **Resilience Pyramid（四层韧性金字塔）：** 01 中等推力保持行走 → 02 极端推力摔倒恢复并恢复行走 → 03 硬件故障下的摔倒恢复与自适应步态（limping / hopping）→ 04 严重损伤转入爬行（crawling）。
- **三阶段训练配方（RL → Distillation → RL）：**
  - **Stage 1（6 个领域专家，PPO + AMP）：** 3 类故障族（Actuation Loss 86 例、Joint Locking 86 例、Knee-Fold Constraint 90°–150° 24 例，共 196 例）× 2 类技能集（Recovery–Locomotion、Crawling），输入含 21 维二值故障指示向量（特权信息）。
  - **Stage 2（DAgger 多教师蒸馏）：** 去掉故障指示器，仅输入速度指令 $(v_x, v_y, \omega_z)$ 与本体感知；对比无记忆 MLP、512 维单层 GRU RNN、64 帧因果 Transformer，Transformer 教师动作 MSE 最低，且 rollout 中 `Other` 模糊帧比例最低（2.6%–9.6% vs MLP 71.3%–79.8%）。
  - **Stage 3（Preference Alignment RL）：** 直立恢复与行走奖励远高于爬行；仅 64 帧 Transformer 成功向直立对齐（直立 42%→76%、爬行 45%→16%、Other 13%→8%），MLP 与 RNN 则塌缩至爬行主导。
- **机制：** 具身 **In-Context Learning** / **Whole-Body Context Learning**——权重冻结下靠 64 帧因果上下文窗口从交互历史推断当前身体与环境约束（含真机膝关节过热掉电再恢复、绑腿解除与高度限制解除后自主站起）。
- **命名：** REACT = **REsilient humAnoid ConTrol**。
- **范式位置：** 亮源新创「规模化预训练 — 规模化对齐 — 规模化部署」中 **部署段** 首个成果（预训练段见 [Light-O1](./light-o1.md)，对齐段见 [LightNav-0](./lightnav-0.md)）。

## 交叉链接

- 微信归档：[wechat_lightorigins_light_react_2026-09-09](../blogs/wechat_lightorigins_light_react_2026-09-09.md)
- 主实体：[Light REACT](../../wiki/entities/light-react.md)
- 同机构：[LightNav-0](./lightnav-0.md)、[Light-Loco-Parkour 项目页](./light-loco-parkour-github-io.md)
