# Awesome-Real2Sim2Real README 通读（技术地图支撑）

> 来源归档（原文通读 · 支撑 `wiki/overview/sun-awesome-r2s2r-technology-map.md`）

- **标题：** Awesome-Real2Sim2Real（sun254667）README
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://github.com/sun254667/Awesome-Real2Sim2Real>（2026-11 通过 raw README.md 获取全文，约 29 KB）
- **一句话说明：** Sim-to-Real / Real-to-Sim / Real2Sim2Real 论文策展清单：每条含出处、要点、链接，末尾有研究挑战表与相关清单。
- **核对范围：** 通读 README 全文：引言与收录标准、目录、7 个分组（综述 3、RL 迁移 6、零样本 8、域随机化与适配 5、基础模型增强 4、视频 Real2Sim 7、操作 Real2Sim 5、GS 闭环 5、框架与流水线 14、基准 7）、「Key Research Challenges」通用挑战与分方向表、Related Awesome Lists、许可。未逐篇打开清单所指向的论文原文（清单性质页，论文各有独立节点页）。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **定位：** 面向机器人 / RL / 具身智能的 Sim-to-Real、Real-to-Sim、Real2Sim2Real 论文精选；许可 CC0-1.0。
- **收录标准：** 优先 CVPR / ICCV / ICLR / NeurIPS / ICML / IROS / ICRA / RA-L 等高质量工作，覆盖 RL 迁移、零样本、域随机化与适配、基础模型增强、视频 Real2Sim、操作 Real2Sim、闭环系统、基准与仿真器。
- **结构：** 综述 -> Sim-to-Real（RL / 零样本 / DR 与适配 / 基础模型）-> Real-to-Sim（视频 / 操作与交互）-> Real2Sim2Real 闭环（Gaussian Splatting / 框架与流水线）-> 基准 -> 研究挑战 -> 相关清单。
- **条目格式：** 每条为 Publication + Highlights + Paper Link；DISCOVERSE、SynthDrive 在 README 中出现两次（分别属闭环 / 框架与基准分组）。
- **通用挑战（4 条）：** 现实差距（动力学 / 感知 / 环境变化）；高保真仿真成本（系统辨识、资产、调参）；泛化（策略过拟合仿真器动力学）；数据稀缺。
- **分方向挑战表：** Sim-to-Real（未建模动力学、视觉策略可迁移性、真机微调样本效率）；Real-to-Sim（从视频恢复物理参数、可变形物体、规模化）；闭环（视觉与动力学双重差距、环路一致性、在线适配）；基础模型集成（VLM 自动系统辨识、生成式数据增强、安全可靠性）。
- **相关清单：** 仅列 AwesomeSim2Real（LongchaoDa）。

## 对 wiki 的映射

- [`wiki/overview/sun-awesome-r2s2r-technology-map.md`](../../wiki/overview/sun-awesome-r2s2r-technology-map.md)
- [`wiki/entities/awesome-real2sim2real.md`](../../wiki/entities/awesome-real2sim2real.md)
