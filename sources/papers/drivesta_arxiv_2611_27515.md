# DriveSTA：Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving

- **类型：** paper / end-to-end autonomous driving
- **标题：** Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving（超越单条人类示范：从带评分的轨迹备选中学习端到端驾驶）
- **用户提供 arXiv：** https://arxiv.org/abs/2611.27515（2026-11-23 尝试访问 abs / html / PDF 均失败，论文身份与版本待从 arXiv 原文复核）
- **项目主页：** https://rtwotwo.github.io/DriveSTA/
- **代码：** https://github.com/Rtwotwo/DriveSTA
- **模型与资产：** https://huggingface.co/Rtwotwo/DriveSTA（官方 README 链接；本次未直接读取该页面）
- **入库日期：** 2026-11-23
- **核对范围：** 官方项目页完整可见技术说明、NAVTEST / HUGSIM 结果表；GitHub 官方 README、docs/CODE_STRUCTURE.md、docs/INFERENCE.md、docs/MINING.md、docs/MODEL_CONFIGS.md、docs/VALIDATION.md、run_inference.py、drivesta_agent.py、drivesta_model.py 的公开内容；**未获取论文全文 / 附录**，因此不对论文公式、作者机构、附录实验和 73% 延迟结论作全文核对。
- **通读：** 未完成

## 论文待核要点与已知线索

1. **动机（官方项目说明）：** 驾驶日志每个场景只给一条已执行的未来轨迹，而规划需要评估多种合理候选。DriveSTA 从原始记录构建可行候选、借助仿真器评分，得到稠密的优质目标；项目页写明 103.3k 真实场景扩展到 2.61M 目标。
2. **阶段一 Proposal（项目页 + 源码）：** 从场景特征生成多个轨迹候选，并利用成功/失败轨迹回放学习评分；官方代码含 DriveSTATrajectoryDecoder 与 ScoreDecoder。
3. **阶段二 Refinement（项目页 + 源码）：** 用共享 JEPA 历史视频表征提供时序上下文，以残差式去噪精修候选，再联合评分与筛选；公开源码在开启 use_diffusion_decoder 时使用 JEPA 和 DiT 相关路径。
4. **NAVSIM v1 NAVTEST（作者项目表）：** DriveSTA Base Proposal / Refinement 为 94.76 / 95.14 PDMS；DriveSTA-Scale Proposal / Refinement 为 95.09 / 95.55 PDMS。注意同项目页复现对比中 FPD-Drive-Pro 为 95.69，不能表述为所有模型中绝对第一。
5. **NAVSIM v2 扩展指标（项目表）：** DriveSTA-Scale Proposal 在 NAVTEST EPDMS 为 91.32，在 NAVHARD EPDMS 为 56.13；与 v1 的 PDMS 不能互换。
6. **HUGSIM 闭环（项目表）：** 共 345 场景，DriveSTA-Scale Refinement 的 RC 56.09、HD-Score 46.69。完整基准需外部 HUGSIM 工具与数据，不能把仓库的 NAVSIM v1 评测脚本当成 HUGSIM 复现。
7. **训练规模（README）：** Base 使用 103,263 真实场景，Scale 使用 103,263 真实 + 129,409 合成 = 232,672 场景；完整归档的 308,119 场景并非全部进 Scale 训练。
8. **时延（用户分享，待核）：** “相比 ChainFlow-VLA 规划前向时延降低 73%”尚未在本次可读取的一手项目表 / README 中找到逐项设置与原始耗时，**暂不作为已核定的客观结果**。

## 论文全文未核对项

- arXiv:2611.27515 的正式标题、作者、机构、上传版本、摘要、实验表与附录；
- 轨迹生成器的数学推导、匹配损失、JEPA 预训练协议与消融；
- ChainFlow-VLA 的对齐测速硬件、batch、是否含预处理/评分及 73% 差异。
- 官方 README 的 arXiv 徽章目前仍是 `arxiv-coming-soon.svg`，与分享中的 arXiv 链接需进一步核对。

## 对 wiki 的映射

- [DriveSTA 独立论文实体](../../wiki/entities/paper-drivesta.md)
- [项目页归档](../sites/drivesta.md)
- [代码仓库归档](../repos/drivesta.md)

**后续：** arXiv 全文可读后逐节核对公式、附录和对比实验，并将“通读”改为“已完成”；此前 PR 保持 Draft。
