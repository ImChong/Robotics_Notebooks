# UniTracker

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 024/161）

- **标题：** UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** Shanghai Jiao Tong Univeristy、Shanghai Artificial Intelligence Laboratory、Shanghai Innovation Institute、Peking University
- **项目页：** https://yinkangning0124.github.io/Humanoid-UniTracker/
- **arXiv：** <https://arxiv.org/abs/2507.07356>
- **GitHub（项目页镜像）：** <https://github.com/yinkangning0124/Humanoid-UniTracker>（训练代码截至 2026-09-18 **未发布**）
- **发表日期：** 2025年9月18日（arXiv v1：2025-07-10）
- **入库日期：** 2026-06-26（161 策展）；**2026-09-18** 深读 ingest 见 [`unitracker_arxiv_2507_07356.md`](unitracker_arxiv_2507_07356.md)
- **一句话说明：** Oracle 特权 PPO → CVAE 在线蒸馏的 G1 通才全身 tracker；deploy 为 25 步本体历史 + 稀疏 goal（**非**相机观测）；partial/full latent 对齐缓解 MLP+DAgger 漂移。深读归纳见 wiki 实体页。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **024/161**。
- **算法实现总结（公众号，已部分过时）：** 早期摘要误写「相机/多视角」；arXiv 正文为特权 Oracle + CVAE 蒸馏，deploy 侧为本体历史 + goal。
- **深读要点（2026-09-18）：** AMASS 11,313 + PHC 过滤 + H2O retarget；Table I Ours SR **91.83** vs DAgger w/o CVAE **88.21**；MDM/GVHMR 下游；**代码未开源**。

## 对 wiki 的映射

- [paper-loco-manip-161-024-unitracker](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
- [unitracker_arxiv_2507_07356.md](unitracker_arxiv_2507_07356.md)
- [humanoid-unitracker-github-io.md](../sites/humanoid-unitracker-github-io.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
