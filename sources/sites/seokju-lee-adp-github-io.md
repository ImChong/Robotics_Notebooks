# ADP 项目页（seokju-lee.github.io/adp）

> 来源归档

- **标题：** ADP: Adversarial Dynamics Priors for Physically Grounded Humanoid Locomotion
- **类型：** site / project-page
- **URL：** <https://seokju-lee.github.io/adp>
- **论文：** <https://arxiv.org/abs/2607.03454> — 归档见 [`sources/papers/adp_arxiv_2607_03454.md`](../papers/adp_arxiv_2607_03454.md)
- **机构：** 韩国科学技术院（KAIST）× 三星电子未来机器人人工智能组 × 汉阳大学 × 韩国机械材料研究院（KIMM）
- **入库日期：** 2026-07-22
- **一句话说明：** ADP 官方项目页：摘要指标卡、方法两阶段示意、仿真四基线对照、G1 真机推扰视频与 BibTeX。

## 公开信息要点（截至 2026-07-22 核查）

| 项 | 状态 |
|----|------|
| **arXiv / PDF** | 已挂：<https://arxiv.org/abs/2607.03454>（页头 Paper / arXiv） |
| **Video** | 页内 Summary Video 与真机/仿真片段 |
| **Code** | 页头按钮文案 **Code (coming soon)**；**无** 可点击训练/推理仓库 URL |
| **GitHub** | [`seokju-lee/adp`](https://github.com/seokju-lee/adp) 为项目站源码（`index.html` + `static/`），**非** 复现仓 |
| **开源结论** | **宣称将开源 / 待发布**（论文亦写 source code will be released on the paper website） |

### 页面内容摘要

- **指标卡：** 四向推扰成功率 **91.4%**（相对 AMP +18 pp）；\(J_{80}\) **+16.7%**；恢复时间 **−47.9%**；速度跟踪误差 **−35.4%**。
- **方法：** (a) SRBD-TO → \(\mathcal{D}_{\mathrm{dyn}}\)；(b) 策略 rollout 动力学时间窗 → 判别器 → \(r_t^D\)。
- **对照：** Vanilla RL / Dynamics Reward / AMP / ADP（同 PPO 设置）。
- **真机：** Unitree G1；冲量推与持续推相对 AMP 的定性对照。
- **消融叙事：** 接触指示消融跌幅最大；时间窗 \(K=8\) 为文中推荐折中。

## 为何值得保留

- 步骤 2.5 项目页核查主入口：用「coming soon」锁定开放边界，避免 wiki 误写「已可复现」。
- 比 PDF 更易浏览的指标与视频入口，便于与 AMP / SD-AMP 选型对照。
- 代码一旦放出，可在此页增量更新并补建 `sources/repos/`。

## 关联资料

- 论文摘录：[`sources/papers/adp_arxiv_2607_03454.md`](../papers/adp_arxiv_2607_03454.md)
- Wiki 实体：[`wiki/entities/paper-adp.md`](../../wiki/entities/paper-adp.md)
