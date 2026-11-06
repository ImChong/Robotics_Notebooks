# GMR：Retargeting Matters（arXiv:2510.02252）

> 来源归档（paper + 官方仓库核对）

- **类型：** paper
- **标题：** Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking
- **arXiv：** <https://arxiv.org/abs/2510.02252>
- **PDF：** <https://arxiv.org/pdf/2510.02252>
- **作者：** Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu（arXiv 摘要页）
- **提交：** 2025-10-02
- **代码：** <https://github.com/YanjieZe/GMR>（git ls-remote 可达，2026-11-05 HEAD `25f56db`；MIT；README 标版本 0.2.0）
- **论文网站：** <https://jaraujo98.github.io/retargeting_matters>（摘要页所列，未打开）
- **入库日期：** 2026-11-05
- **线索来源：** [wechat_yuanxq_retargeting_general_tracking_2026-11-05.md](../blogs/wechat_yuanxq_retargeting_general_tracking_2026-11-05.md)
- **核对范围：** 仅 arXiv 摘要页、仓库 README 与 `scripts/smplx_to_robot.py`、`general_motion_retargeting/motion_retarget.py` 入口；论文全文未通读。
- **通读：** 未完成

## arXiv 摘要要点（人工转述）

摘要称：作者系统评估重定向质量如何影响动作跟踪策略（在抑制过度奖励调参的条件下），指出脚滑、自穿透和物理不可行等重定向伪影常被留给 RL 策略去纠正；提出 GMR，并与 PHC、ProtoMotions 两个开源重定向器及 Unitree 的闭源高质量数据集比较；用 BeyondMimic 训练策略，在 LAFAN1 子集上称 GMR 在跟踪表现与对源动作的忠实度上持续优于现有开源方法，接近闭源基线。

## 命名与易混点

- 官方 README 同时挂了两个 arXiv 徽章：2505.02833 是 TWIST（Teleoperated Whole-Body Imitation System，见 [paper-twist](../../wiki/entities/paper-twist.md)），GMR 本体技术报告是 2510.02252。
- 仓库 `engineai-robotics/GMR` 等厂商分支见 `cn-os-*-gmr` 页，不是本条目。

## 对 wiki 的映射

- [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)
