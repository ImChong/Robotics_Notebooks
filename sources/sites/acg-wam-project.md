# ACG-WAM 项目页

> 来源归档（site；2026-10-08 通过官方仓库 README 与论文页核查）

- **项目页：** <https://RoboOpus.github.io/ACG-WAM/>
- **论文：** <https://arxiv.org/abs/2610.06965>
- **代码：** <https://github.com/RoboOpus/ACG-WAM>
- **权重：** <https://huggingface.co/RoboOpus/ACG-WAM>
- **关联资料：** [论文摘录](../papers/acg_wam_arxiv_2610_06965.md)；[代码仓](../repos/acg-wam.md)
- **对 wiki 的映射：** [ACG-WAM](../../wiki/entities/paper-acg-wam-geometric-latent-prediction.md)

## 项目页与开放状态核查

- 项目页 README 指向模型卡；检查点名为 **ACG-WAM-RoboTwin-40K**，按 40k updates 的 Joint ACG-JEPA 配方训练。
- 页面称 RoboTwin 2.0 50-task mean SR 为 93.07%；真机结果为 TRON2 + WUJI hands 三项任务平均 SR 85.00%、PCS 91.67%。
- 训练实现已公开，许可证 Apache-2.0；项目代码的公开 RoboTwin policy 路径不等于其公开 TRON2 真机部署栈。
- 权重文件约 16.09 GB；权重不带完整训练恢复所需的 optimizer/scheduler state。
- 仓库未附演示数据或 VGGT teacher cache；训练仍需单独准备 RoboTwin 数据与 Wan / Qwen / Motus / VGGT 上游资产。
