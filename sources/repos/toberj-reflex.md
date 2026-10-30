# ToberJ/reflex（Reflex 官方仓库）

> 来源归档（官方仿真实现）

- **链接：** <https://github.com/ToberJ/reflex>
- **论文：** <https://arxiv.org/abs/2610.12772>
- **核查日期：** 2026-10-30
- **状态：** 已公开三阶段仿真训练、评测代码和 checkpoint；真机部署栈未包含。
- **环境：** Isaac Sim 5.1.0、Isaac Lab 2.3.1、rsl-rl-lib 3.0.1；视觉评测 320 环境约需 14 GB 显存。
- **数据：** 参考动作需申请 BONES-SEED 并使用 `tools/motion/build_reference_motion.py` 重建；训练还需 AMP 动作数据。

## 已核查入口

- `scripts/eval.sh visual` 加载评测配置、latent 控制器和视觉 checkpoint，调用 `reflex.eval.per_throw_eval` 输出逐次抛掷 JSON；`true_z` 和 `teacher` 为对照。
- `scripts/train_stage3.sh` 顺序调用热图训练、特征网络构建、teacher rollout 特征采集、时序编码器训练和 DAgger，再运行视觉评测。
- `source/reflex/reflex/vision/detector_model.py` 中 `DetectorKPSeq` 以 RGB-D 特征及相机自运动输入 GRU，输出 latent 与辅助状态预测。
- 权重下载与训练步骤见官方 README；复现依赖及许可见仓库说明。

## 与知识库的关系

- [Reflex 论文实体页](../../wiki/entities/paper-reflex-visually-grounded-reactive-humanoid-control.md)
- [论文来源归档](../papers/reflex_visually_grounded_reactive_humanoid_control.md)
