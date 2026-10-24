# OmniInteract（arXiv:2610.26043）

> 来源归档（paper）

- **标题：** OmniInteract: Towards Unified Whole-Body Interaction Control for Dual Humanoids
- **类型：** paper / humanoid / dual-humanoid-interaction / whole-body-control
- **arXiv：** <https://arxiv.org/abs/2610.26043>
- **HTML：** <https://arxiv.org/html/2610.26043v2>
- **项目页：** <https://calvinyhli.github.io/omniinteract.github.io/>
- **作者：** Yinghao Li、Tianyu Zhang、Zhuohao Li、Yi-Lin Wei、Fa-Ting Hong、Qing Zhang、Wei-Shi Zheng
- **机构：** 中山大学计算机科学与工程学院、深圳河套学院、北京理工大学、香港科技大学
- **版本：** v1 于 2026-10-19 提交，v2 于 2026-10-20 修订
- **入库日期：** 2026-10-24
- **一句话说明：** 用表面接触感知重定向保留两台人形机器人的接触界面，再以冻结的全身运动先验和流匹配残差策略扩展双人形交互技能。

## 核心摘录

1. **SCAR 重定向：** Surface Contact-Aware Retargeting 联合优化两台机器人，将稠密人体—机器人表面对应、双向接触界面约束和接触附近的几何关系纳入运动重定向；逐帧以 SQP 求解并考虑关节、速度、地面与碰撞约束。
2. **可扩展交互控制：** 冻结 SONIC 全身跟踪先验；针对每类交互，用 MAPPO 学习依赖伙伴状态的残差专家。之后通过 DAgger 收集专家残差监督，以 conditional flow matching 合并为统一策略，部署时把生成的残差叠加到 SONIC 动作上。
3. **评测与真机：** 在 InterHuman、Inter-X 人体交互动作上评测重定向和控制；论文报告仿真未见动作成功率 72.89%，训练 60 种动作时成功率 84.3%。两台 Unitree G1 的 High-Five、Hug、Shoulder 真机任务成功率分别为 85.7%（14 次）、72.7%（22 次）、77.8%（18 次）。
4. **开源状态：** 截至 2026-10-24，论文与官方项目页提供论文、交互演示和完整视频；项目页未列出 GitHub、Hugging Face 或数据下载链接，暂无可运行官方源码入口。

**对 wiki 的映射**

- [OmniInteract 双人形全身交互控制](../../wiki/entities/paper-omniinteract-dual-humanoids.md)
- [人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md)
