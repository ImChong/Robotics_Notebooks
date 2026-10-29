# Paper: LCP (2024)
- **Title**: Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies
- **Authors**: Zixuan Chen, Xialin He, Yen-Jen Wang, Qiayuan Liao, Yanjie Ze, Zhongyu Li, S. Shankar Sastry, Jiajun Wu, Koushil Sreenath, Saurabh Gupta, Xue Bin Peng
- **arXiv**: <https://arxiv.org/abs/2410.11825>（v1 2024-10-15，v3 2024-10-28；cs.RO）
- **核对日期**: 2026-10-26（此前误记为 SIGGRAPH 2023 / Physics-Based Character Control，已按 arXiv 页面更正）
- **要点（据摘要）**: 低通滤波与平滑奖励不可微且需逐机器人调参；LCP 把 Lipschitz 约束实现为可微的梯度惩罚，可替代平滑奖励与滤波器，并在多款人形机器人的仿真与真机上验证。代码与 checkpoint 在项目页发布。
