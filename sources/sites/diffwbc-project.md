# OneSteer / DiffWBC 官方项目主页归档

> 来源归档（site）；同一篇 arXiv:2610.41354，v1 名称 DiffWBC，v3 正式名称 OneSteer。

- **英文标题：** OneSteer: Unlocking Steerable General Humanoid Motion Tracking via One-Step Diffusion Whole-Body Control
- **中文标题：** OneSteer：通过单步扩散全身控制实现可引导的通用人形运动跟踪
- **类型：** site / project / humanoid-motion-tracking
- **项目页：** https://onesteer.github.io/
- **论文：** https://arxiv.org/html/2610.41354v3
- **预印本：** 2026-10-30 v1（DiffWBC）；2026-11-11 v3（OneSteer）
- **首次归档：** 2026-11-10；**最新核查：** 2026-11-14
- **一句话说明：** 在 Unitree G1 上将单步状态/动作计划、参考分组遮蔽、可微代价 steering 与失效片段筛查纳入一个闭环控制器。
- **核对范围：** 官方首页 Abstract、Real robot、Steering、Terrain、Method、Results、Self-assessment、Paper / Code 入口及页尾；与 arXiv:2610.41354v3 及旧版 v1 核对标题与实验结果。
- **通读：** 已完成

## 开放状态（2026-11-14）

- **论文：已公开**（最新 v3，网页 Paper 链接可直达 arXiv）。
- **训练 / 推理 / 部署代码：未开放。** 项目页显示「Code (coming soon)」，未给出可运行官方 GitHub、权重或部署脚本；不可把公开视频当作源码开放，源码运行时序图不适用。
- **数据：** 论文说明训练使用 BONES-SEED、AMASS 与 LAFAN，但项目页未给出该项目自己的整理数据下载链接。
- **论文身份：** v1 的 DiffWBC 与 v3 的 OneSteer 同一 arXiv ID、同一研究成果，仓库保留一个论文 wiki 节点 [paper-diffwbc](../../wiki/entities/paper-diffwbc.md)。

## 演示与评测核对（官网 / v3）

1. **普通跟踪：** 50 Hz 预测闭环 G1 控制，ONNX 模型使用 Orin NX，9 条动作 × 3 次真机普通跟踪全部未摔倒。单次生成器 RTX 4090 耗时 3.8 ms。
2. **真机 steering：** 使用机载 LiDAR 构建的、实验前固定 SDF 地图，冻结同一个策略权重；1.1 m 低杆 4/5 通过，12 cm 跨栏 3/3 未摔倒。**这些 steering 实验用系留 RTX 4090 计算**，不代表已在 Orin NX 端到端运行。
3. **仿真 steering：** 79 条未见演员 walking 窗口、1.25 m 低杆 clean pass 从 0.02 提升至 0.90（只优化第 2 次生成器调用的输入、两次梯度下降）；闭环编辑原参考最高 0.11。
4. **地形扩展：** 从零模型在 21×21 地形扫描→36 个 tokens 基础上 RL 微调；6 类地形上的整体成功率 0.43→0.90，**目前只有仿真**。
5. **失效筛查：** K=8 个生成噪声计算预测计划离散度，top 2% 子窗口评分 precision 0.91；用 329 个困难片段微调修复 48±4% 失败。
6. **版本口径：** v1 150 条 BONES 窗口、20.1 mm，不同于 v3 **512 条、22.6 mm**；v1 页面旧结论「真机没有 steer」已被 v3 真机实验更新。引用必须携带版本号。

## 对 wiki 的映射

- [唯一论文实体：OneSteer（旧名 DiffWBC）](../../wiki/entities/paper-diffwbc.md)
- [OneSteer v3 论文通读归档](../papers/onesteer_arxiv_2610_41354_v3.md)
- [DiffWBC v1 历史通读归档](../papers/diffwbc_arxiv_2610_41354_reading.md)

## 参考来源

- https://onesteer.github.io/
- https://arxiv.org/html/2610.41354v3
- https://arxiv.org/html/2610.41354v1
