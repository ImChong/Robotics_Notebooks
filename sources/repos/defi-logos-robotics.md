# defi-logos-robotics

> 来源归档（ingest）

- **标题：** LogosRoboticsGroup/DeFi — Disentangled Robot Learning 官方实现
- **类型：** repo
- **链接：** <https://github.com/LogosRoboticsGroup/DeFi>
- **权重：** <https://huggingface.co/zbzzbz/DeFI>
- **论文：** <https://arxiv.org/abs/2604.16391>
- **入库日期：** 2026-05-20
- **一句话说明：** DeFI（GFDM + GIDM + 扩散动作适配器）的训练/评测代码与 checkpoint 发布入口，对应 arXiv:2604.16391。

## 仓库要点（README 级）

- **模块边界：** 前向动力学（视频生成 / SVD 系）、逆动力学（自监督潜动作 + VQ）、下游耦合微调与 CALVIN / SimplerEnv / 真机评测脚本（以仓库当前目录为准）。
- **依赖生态：** 与论文一致的扩散视频骨干、DINOv2、T5 指令编码、DiT 动作适配器等；具体环境与数据准备见官方 README。
- **复现注意：** 多视角（第三人称 + 腕部）在 CALVIN 设定下与论文 Table 1 两行对照相关；真机 Franka 八任务协议见论文 §4.4。

## 关联 Wiki 页面

- [DeFI（解耦前向/逆动力学 VLA）](../../wiki/methods/defi-decoupled-dynamics-vla.md)
- [VLA](../../wiki/methods/vla.md)

## 当前提炼状态

- [x] 官方链接与论文映射
- [ ] 后续：随仓库 README 更新补充训练命令与 checkpoint 命名
