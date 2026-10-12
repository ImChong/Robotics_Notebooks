---
type: entity
project_id: clotho-loose-inertial-garments
doi: "10.1145/3842534"
tags: [paper, human-motion, mocap, imu, wearable-sensing, motion-tracking, siggraph-asia, acm, xiamen-university, mit, cmu]
status: complete
updated: 2026-10-12
venue: "ACM Transactions on Graphics · SIGGRAPH Asia 2026"
project: https://clotho-mocap.github.io/
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ./paper-egoexomocap.md
  - ./paper-notebook-avatarposer-articulated-full-body-pose-tracking.md
  - ./paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md
  - ./paper-humaps4d.md
sources:
  - ../../sources/papers/clotho.md
  - ../../sources/sites/clotho-mocap.md
summary: "CLOTHO（TOG / SIGGRAPH Asia 2026，DOI 10.1145/3842534）面向松散穿戴惯性服装：Masked IMU Encoder + 条件残差 canonicalizer 将异构稀疏 IMU 先校正为标准 10-IMU 表示，再做全身运动恢复；项目页报告 116 小时、5 种服装、对 FIP/LIP 位置误差降低 42% 与 8 小时跟踪；代码和数据尚无可核实下载入口。"
---

# CLOTHO：先校正松散 IMU，再恢复人体运动

**CLOTHO: Canonicalizing IMUs from Loose Inertial Garments for Accurate Human Motion Tracking**（Gao et al., ACM Transactions on Graphics / SIGGRAPH Asia 2026，DOI [10.1145/3842534](https://doi.org/10.1145/3842534)，[项目主页](https://clotho-mocap.github.io/)）针对宽松惯性服装里「传感器跟着衣服走，不一定跟着身体走」的问题。它在姿态重建前增加一个 **IMU canonicalization** 阶段：把受衣物滑移、材料、贴合度与布局影响的稀疏观测，映射到漂移校正、空间完整的标准 10-IMU 信号，再交给下游模型重建全身运动。

## 一句话定义

CLOTHO 将穿戴域适配从人体姿态估计中拆出，以配置掩码编码部分 IMU，再预测残差校正到统一的 10 传感器表示，提升不同服装和传感器配置下的惯性动捕精度与稳定性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMU | Inertial Measurement Unit | 穿戴式惯性测量单元，提供方向/运动线索 |
| OME | Orientation Mean Error | 姿态角度误差；项目页用于报告数据规模消融 |
| FIP | Full-Body Inertial Poser | 项目页对比的全身惯性姿态估计基线 |
| LIP | Learned Inertial Poser | 项目页对比的学习式惯性姿态估计基线 |
| BNO085 | Bosch Sensortec BNO085 IMU | 三套 10 传感器服装采用的低成本 IMU |
| MTi-3 | Xsens MTi-3 | Purple-6 服装所用的惯性传感器型号 |

## 为什么重要

- **把衣物扰动和姿态重建分开处理：** 传统做法往往让 pose 网络直接适配传感器位置偏移。CLOTHO 的中间 canonical representation 使模型可以先处理“穿戴配置是什么、信号偏了多少”，再做人体运动恢复。
- **针对真实部署里的服装差异：** 传感器数目、布置位置、材料与服装松紧都会变；模型对未见服装配置零样本泛化，目标是降低每换一件衣服就要重新标定或训练的负担。
- **与稀疏可穿戴动捕路线互补：** [AvatarPoser](./paper-notebook-avatarposer-articulated-full-body-pose-tracking.md) 从 HMD/手柄稀疏输入补全人体姿态；[EgoExoMoCap](./paper-egoexomocap.md) 将分布式 HMD 的 ego / exo 观测融合。CLOTHO 的研究重点更靠近惯性服装的传感器校正和跨穿戴配置稳健性。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 厦门大学（Xiamen University）、麻省理工学院（MIT）、卡迪夫大学（Cardiff University）、卡内基梅隆大学（Carnegie Mellon University） |
| **论文** | ACM Transactions on Graphics（SIGGRAPH Asia 2026），DOI `10.1145/3842534` |
| **数据规模** | 项目页报告约 12.5M IMU 帧、约 116 小时、5 种服装设计 |
| **传感器配置** | Green-10 / Taichi-10 / White-10：10×BNO085；Purple-6：6×Xsens MTi-3；T-Shirt-4：4×Noitom PN3 |
| **项目页报告** | 五种服装上位置误差最低；相对 FIP/LIP 位置误差降低 42%；展示 8 小时连续跟踪 |
| **开源状态** | 项目页宣称软件、权重、硬件与数据将开放；截至 2026-10-12 未能核实可访问的代码仓或数据下载 URL |

## 方法

```mermaid
flowchart LR
  loose["松散服装 IMU 序列"] --> mask["传感器配置掩码"]
  loose --> enc["Masked IMU Encoder"]
  mask --> enc
  enc --> latent["配置不变 latent"]
  latent --> canon["Latent-Conditioned IMU Canonicalizer"]
  loose --> canon
  canon --> standard["残差校正的标准 10-IMU 表示"]
  standard --> recon["下游运动重建器"]
  recon --> pose["全身人体运动"]
```

### 1）配置不变的编码

输入可能缺少标准身体位置的若干传感器。**Masked IMU Encoder** 结合有效传感器掩码学习 latent，使编码尽量不依赖具体服装配置。掩码提供“哪些通道存在”的结构信息，避免把缺测信号和真实静止混淆。

### 2）条件残差 canonicalization

**Latent-Conditioned IMU Canonicalizer** 读取原始 IMU 序列和上述 latent，预测从当前松散穿戴观测到标准表示的残差修正。输出是固定的 10-IMU 空间配置，同时校正传感器相对身体的偏移与长期漂移。之后才由下游人体运动重建模块估计姿态。

这个分解的关键在接口：CLOTHO 先输出规范化后的传感器运动表征，而不是直接把所有穿戴差异压进最终人体姿态回归器。因而该 canonicalizer 可以作为面向 IMU 服装的前置层来理解。

## 评测

### 跨服装数据

项目页报告共采集 **约 116 小时、12.5M 帧**，覆盖五种在材料、版型、IMU 数量和硬件上不同的惯性服装。三种 10-IMU 服装采用约 US$19/个的 BNO085；Purple-6 采用 Xsens MTi-3；T-Shirt-4 采用 Noitom PN3。测试覆盖未见服装与未见传感器数量配置，检验 canonicalization 对穿戴域变化的泛化。

### 项目页公布的主要结果

- **相对惯性姿态基线：** 项目页称 CLOTHO 在所有测试服装上的位置误差最低，较 FIP / LIP 的位置误差降低 **42%**。这是项目页给出的汇总描述；由于本条目无法直接核对可下载论文全文中的表格定义、拆分协议和置信区间，横向引用时应保留这一来源限定。
- **长时稳定性：** 项目页展示最长 **8 小时**跟踪，用于说明 canonicalization 对漂移的抑制；这不代表任意硬件、穿戴方式或实时部署条件下都能保证该时长。
- **标签与数据规模：** 7.07 小时配对标注使 OME 从 **16.05° 降至 13.78°**。项目页还报告真实 0.55 小时配对数据优于 7 小时合成服装数据；无标签 IMU 数据扩至 101.1 小时仍有收益，表明 paired supervision 与无标签规模作用互补。
- **模型容量：** 页面报告模型从 10.1M 扩到 92M 参数仅带来约 1.02° OME 改善；在该实验设定中，继续扩大数据比增加模型容量更值得优先验证。

## 对比

| 路线 | 主要输入与困难 | CLOTHO 的区别 |
|------|----------------|----------------|
| [AvatarPoser](./paper-notebook-avatarposer-articulated-full-body-pose-tracking.md) | HMD 与控制器等稀疏位姿，重点是从少量点补全人体关节 | CLOTHO 面向身体 IMU 服装，并先标准化传感器信号 |
| [EgoExoMoCap](./paper-egoexomocap.md) | 多 HMD 的 ego 轨迹与 exo RGB，重点是分布式视角融合 | CLOTHO 处理衣物运动造成的 IMU / 身体耦合变化 |
| 传统 IMU pose 回归 | 给定固定传感器位置，直接由惯性序列预测 pose | CLOTHO 在 pose 网络前显式校正到统一 10-IMU 表示 |

## 源码运行时序图

**不适用：** 截至 2026-10-12，项目主页虽然声明将开放软件和权重，但尚未核实到可访问的官方代码仓、安装说明或训练/推理入口；当前没有依据绘制可复现的代码运行时序。

## 工程实践

- **复现前先确认资源落地：** 项目页的 Code / resources 入口目前没有可核实的仓库 URL；下载训练脚本、模型权重和五套服装数据的状态尚不明确。实际复现前应重新检查[官方项目页](https://clotho-mocap.github.io/)及 DOI 关联的正式版本。
- **分开比较两类误差：** 位置误差与 OME 不是同一指标。位置误差下降 42% 的对比和数据规模消融里的 OME 数字应分开引用。
- **部署仍需验证硬件匹配：** 低成本 BNO085、Xsens MTi-3 与 Noitom PN3 涉及不同硬件与数量；跨设备的标定、同步、坐标系转换和实时延迟不能仅由模型跨服装结果推出。
- **源码运行时序图：不适用**（官方可运行实现尚未核实）。

## 局限与风险

- **当前复现入口不完整：** 项目页宣称会开放软件、权重、硬件和数据，但本条目核查时没有可下载的代码或数据链接；数据许可、预处理、训练细节、真实数据划分与硬件清单仍需公开材料补足。
- **项目页指标需论文表格核验：** 42% 汇总降幅、8 小时跟踪与零样本泛化是项目方报告。实际部署判断还需核对正式论文对误差定义、样本/被试拆分、穿戴校准及长时段测试条件的说明。
- **标准化仍依赖目标定义：** canonicalizer 的目标是固定 10-IMU 表示；对完全不同穿戴位置、传感器品质或未覆盖动作分布的效果需要单独评测。
- **模型与数据的投入权衡来自特定实验：** 页面报告扩模型收益有限、扩数据仍有收益，不应直接外推到不同架构或应用场景。

## 结论

CLOTHO 的主要贡献是把松散惯性服装的穿戴误差显式变成一个可学习的 canonicalization 问题，先恢复稳定的标准 IMU 表示，再进行全身姿态重建。

1. **真正应关注的指标：** 同时看位置误差、OME、跨服装/传感器数量的零样本表现和长时漂移；单一平均 pose 指标不能体现穿戴鲁棒性。
2. **方法价值：** 配置掩码 + 条件残差校正提供了清晰的模块边界，能区分“传感器挂在衣服上如何偏了”和“人体现在是什么姿势”。
3. **数据策略：** 项目页结果提示真实少量配对样本与大量无标签 IMU 有互补作用；复现应分别报告两种数据贡献。
4. **硬件代价：** 五套服装覆盖多传感器配置，但部署仍涉及传感器同步、校准、服装维护与对齐；论文中的算法误差不等于端到端穿戴成本。
5. **当前复现判断：** 项目页承诺开放，实际代码/数据链接待核实；在资源公开前，可把它作为方法与评测参考，不宜写成已可复现基线。

## 参考来源

- [论文摘录与 DOI](../../sources/papers/clotho.md)
- [CLOTHO 项目页归档](../../sources/sites/clotho-mocap.md)
- [官方项目主页](https://clotho-mocap.github.io/)
- [ACM DOI](https://doi.org/10.1145/3842534)

## 关联页面

- [Human Motion 论文索引](../overview/paper-notebook-category-14-human-motion.md)
- [EgoExoMoCap](./paper-egoexomocap.md) — ego / exo 多视角稀疏人体动捕
- [AvatarPoser](./paper-notebook-avatarposer-articulated-full-body-pose-tracking.md) — HMD 稀疏输入人体姿态估计
- [HUMAPS-4D](./paper-humaps4d.md) — 多模态人体运动数据采集

## 推荐继续阅读

- [FIP: Full-Body Inertial Poser](https://fangjw-0722.github.io/) — 对照惯性姿态重建基线
- [CLOTHO 项目主页](https://clotho-mocap.github.io/) — 后续核对代码、数据与模型发布状态
