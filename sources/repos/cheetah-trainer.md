# Chenaah/Cheetah-Trainer（+ Cheetah-Gym / Cheetah-Software-RL，ARRL 多模态腿足代码）

> 来源归档

- **标题：** Cheetah-Trainer / Cheetah-Gym / Cheetah-Software-RL
- **类型：** repo
- **来源：** ShanghaiTech University（Chen Yu，GitHub id: Chenaah）
- **链接：** <https://github.com/Chenaah/Cheetah-Trainer> · <https://github.com/Chenaah/Cheetah-Gym> · <https://github.com/Chenaah/Cheetah-Software-RL>
- **项目页：** <https://chenaah.github.io/multimodal/> — 归档见 [`sources/sites/multimodal-chenaah-github-io.md`](../sites/multimodal-chenaah-github-io.md)
- **入库日期：** 2026-07-28
- **一句话说明：** Multi-Modal Legged Locomotion（ARRL，RA-L/IROS 2022）官方代码三仓：PyBullet 仿真环境、TensorFlow 训练/测试代码、真机 C++ 程序，另附支撑结构 STL。
- **沉淀到 wiki：** [`wiki/entities/paper-multimodal-legged-arrl.md`](../../wiki/entities/paper-multimodal-legged-arrl.md)

---

## 核心定位

论文项目页 Code 区列出的官方实现，覆盖「仿真环境 → 训练 → 真机部署」全链路：

| 仓库 | 角色 | 技术栈 |
|------|------|--------|
| Cheetah-Gym | PyBullet 仿真环境（Mini Cheetah 双足模式 + 支撑结构） | Python / PyBullet |
| Cheetah-Software-RL | ARRL 训练/测试（TD3/SAC × ES/BO 组合） | TensorFlow / Python |
| Cheetah-Trainer | Mini Cheetah 真机程序 | TensorFlow / C++ |

- **附带资产：** 支撑结构（3D 打印 stick）STL 文件，对应论文机械设计。
- **复现注意：** TF1 时代训练栈；真机部分依赖 Cheetah-Software 系 MIT 控制栈改造，部署前需核对电机/通信接口与自家平台一致性。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-multimodal-legged-arrl](../../wiki/entities/paper-multimodal-legged-arrl.md) | 本仓库对应的论文实体页 |
| [paper-residual-rl-robot-control](../../wiki/entities/paper-residual-rl-robot-control.md) | ARRL 的思想源头（Residual RL），并把「base 控制器手调」进一步自动化 |
