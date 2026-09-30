# amazon-far/PRISM-Real2Sim2Real

> 来源归档

- **标题：** PRISM-Real2Sim2Real（官方实现，项目页链接）
- **类型：** repo
- **代码：** <https://github.com/amazon-far/PRISM-Real2Sim2Real>
- **论文：** [arXiv:2609.38172](https://arxiv.org/abs/2609.38172)
- **项目页：** <https://prism-real2sim2real.github.io/>
- **入库日期：** 2026-09-30
- **最近复核：** 2026-09-30
- **一句话说明：** PRISM Real2Sim2Real 官方 GitHub（Amazon FAR）；项目页已挂 **Code** 入口；截至入库日 **404 Not Found**，README/目录待公开后补全。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-prism-real2sim2real.md`](../../wiki/entities/paper-prism-real2sim2real.md)

## 开源核查（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **待发布 / 待核实** | 项目页 Footer 有 Code 链；匿名访问 GitHub 与 API 均为 **404**（2026-09-30）。论文 §4 写「refer readers to our **codebase**」，预期将公开。 |

公开后应在此补：默认分支、重建脚本入口、MuJoCo/Isaac 训练命令、与 CRISP/SAM 子模块关系。

## 预期运行时模块（论文叙述，非 README 验证）

1. V2V / 数据准备（SeedDance 2.0 生成 counterfactual clips）
2. Contact-anchored Real2Sim（CRISP 扩展 + SAM2/SAM3D + 接触相位）
3. Contact-anchored retargeting → 仿真轨迹
4. Co-tracking teacher RL → DAgger+PPO depth student
5. 真机：Fast-FoundationStereo 深度 + G1 50 Hz 部署
