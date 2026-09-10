---
type: entity
tags: [tool, simulation, connectomics, drosophila, flybrainlab, neuroscience, open-source, columbia]
status: complete
updated: 2026-09-10
related:
  - ../concepts/fly-connectomics-stack.md
  - ./flywire.md
  - ./male-cns-connectome.md
  - ./neuprint.md
sources:
  - ../../sources/repos/flybrainlab.md
summary: "果蝇脑可执行回路交互平台：从连接组数据 3D 探索、编译神经回路到 Neurokernel GPU 仿真，eLife 2021 发表。"
---

# FlyBrainLab

**FlyBrainLab** 是 Fruit Fly Brain Observatory 团队开源的 **交互计算平台**（https://github.com/FlyBrainLab/FlyBrainLab），用于从果蝇脑连接数据 **探索、构建并仿真可执行神经回路**。它把 [FlyWire](./flywire.md) / [neuPrint](./neuprint.md) 等提供的 **结构布线** 推进到 **功能逻辑检验**，论文发表于 *eLife* 2021（[10.7554/eLife.62362](https://dx.doi.org/10.7554/eLife.62362)）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | Neurokernel 回路仿真加速 |
| API | Application Programming Interface | 前后端通信接口 |
| CNS | Central Nervous System | 中枢神经系统 |
| EM | Electron Microscopy | 底层连接组成像模态 |
| ROI | Region of Interest | 回路元素空间范围 |

## 为什么重要

- **结构→功能桥梁：** 连接组回答布线；FlyBrainLab 让实验者 **交互调参** 检验回路是否产生预期动力学。
- **计算/实验分工：** 计算研究者配置可执行回路，神经科学家在 UI 中探索而无需写仿真代码。
- **与 Male CNS 互补：** 雄性全 CNS 新布线可导入同类工作流，研究 **性别特异回路** 的功能后果。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 哥伦比亚大学（Columbia University）Fruit Fly Brain Observatory |
| 仓库 | https://github.com/FlyBrainLab/FlyBrainLab |
| 论文 | Lazar et al., *eLife* 2021 |
| Docker | `fruitflybrain/fbl` |
| 安装 | `pip install flybrainlab[full]`（conda 环境 Python 3.9） |

## 流程总览

```mermaid
flowchart LR
  EXP[3D 探索 NeuroNLP/BrainMapsViz]
  EXP --> CIR[从连接数据编译回路]
  CIR --> SIM[Neurokernel GPU 仿真]
  SIM --> VAL[交互验证功能逻辑]
```

## 工程实践

**仅可视化（最快）：**

```bash
conda create -n flybrainlab python=3.9 git -y
conda activate flybrainlab
python -m pip install flybrainlab[full] neuromynerva
```

默认连接 **公共后端**（无 GPU 回路执行）。亦可试用 [BrainMapsViz](http://www.fruitflybrain.org/#/brainmapsviz)。

**完整 GPU 仿真：** Docker `fruitflybrain/fbl` 或 AWS AMI `ami-02218ae5a3d1fd06d`，本地启动 Neurokernel 后端。

## 局限与风险

- **公共后端能力受限：** 仅用户端安装 **不能** 在远程 GPU 上跑回路；需完整部署。
- **依赖链复杂：** `autobahn-sync`、`Neuroballad`、`nxcontrol` 等 git 依赖，环境冲突时需查 [Troubleshooting wiki](https://github.com/FlyBrainLab/FlyBrainLab/wiki/Troubleshooting)。
- **与机器人栈距离：** 面向 **神经科学验证**，非直接机器人控制中间件。

## 关联页面

- [果蝇连接组工具栈](../concepts/fly-connectomics-stack.md)
- [FlyWire](./flywire.md)
- [Male CNS Connectome](./male-cns-connectome.md)

## 参考来源

- [FlyBrainLab 仓库归档](../../sources/repos/flybrainlab.md)

## 推荐继续阅读

- [FlyBrainLab eLife 2021 论文](https://dx.doi.org/10.7554/eLife.62362)
- [FlyBrainLab Tutorials](https://github.com/FlyBrainLab/Tutorials/)
