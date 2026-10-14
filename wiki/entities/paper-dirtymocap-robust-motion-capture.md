---
type: entity
tags: [paper, motion-capture, human-motion-reconstruction, smpl-h, differentiable-optimization, zju, westlake, fudan, nudt]
status: complete
updated: 2026-10-14
arxiv: '2609.19927'
venue: SIGGRAPH Asia 2026
project: https://wanglongzju.github.io/DirtyMoCap-Project-Page/
code: https://github.com/WangLongZJU/DirtyMoCap
related:
  - ../concepts/motion-retargeting-pipeline.md
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ./paper-mamma-markerless-motion-capture.md
  - ./paper-humantracker.md
sources:
  - ../../sources/papers/dirtymocap_arxiv_2609_19927.md
  - ../../sources/sites/dirtymocap.md
  - ../../sources/repos/dirtymocap.md
summary: DirtyMoCap 将稀疏、含噪、无序的光学 marker 映射为固定身体代理锚点，再用可微 Gauss–Newton 拟合 SMPL-H；单模型跨 marker 布局重建人体动作。作者报告最高 100× CUDA 求解器加速，但官方代码仍待发布，HKMALA-Motion 数据也尚未开放下载。
project_id: dirtymocap-robust-motion-capture
---

# DirtyMoCap: Robust Motion Capture from Unconstrained Markers

## 一句话定义

DirtyMoCap 是一套 marker-layout-free 光学动捕重建方法：把稀疏、噪声、身份/顺序不稳定的 marker 点转成固定身体代理锚点，再恢复时序一致的 SMPL-H 姿态、全局平移与体型。

| 项目 | 内容 |
|---|---|
| 作者 | Long Wang、Shuting Zhao、Shen Yan、Siyuan Yu、Xiaoben Li、Zeyu Cai、Yumeng Hou、Yuliang Xiu |
| 机构 | 浙江大学、西湖大学、复旦大学、国防科技大学、南京大学、新加坡国立大学 |
| 论文/会议 | arXiv:2609.19927（2026-09-17）；官方项目页列 SIGGRAPH Asia 2026 |
| 官方代码 | [WangLongZJU/DirtyMoCap](https://github.com/WangLongZJU/DirtyMoCap) 当前为待发布占位仓库 |
| 项目数据 | HKMALA-Motion 页面标注 Coming soon；目前未确认可下载数据 |

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 光学动作捕捉；本方法输入为可能损坏或无序的 marker 轨迹 |
| SMPL-H | Skinned Multi-Person Linear Model with Hands | 带手部参数的人体模型，输出姿态、平移和体型 |
| CUDA | Compute Unified Device Architecture | 自定义高性能核用于运动学链 Jacobian 计算 |
| MPJPE | Mean Per Joint Position Error | 关节平均位置误差，单位毫米 |
| MPVPE | Mean Per Vertex Position Error | SMPL-H 网格顶点平均位置误差，单位毫米 |
| SSM | Synchronized Scans and Markers | 论文用于真实采集评估的同步人体扫描与 marker 数据 |

## 为什么重要

传统光学 MoCap 常依赖特定 marker 贴法、marker 身份/顺序和人工清洗。遮挡后 marker 重新出现可能换序，ghost marker 和噪声又会破坏模型拟合；跨任务复用采集数据时，固定布局先验尤其脆弱。DirtyMoCap 把对应关系前置到稳定的身体代理锚点，让同一模型处理多种布局，对动作库整理、人体网格恢复和后续重定向有用。

## 核心原理

### 流程总览

```mermaid
flowchart LR
  raw["稀疏、噪声、无序 markers"] --> init["首帧代理锚点初始化"]
  init --> track["递归滑窗锚点轨迹预测"]
  raw --> track
  track --> solver["可微、可学习 Gauss-Newton 拟合"]
  solver --> smpl["SMPL-H 姿态、平移、体型"]
  smpl --> dataset["下游动作标准化 / 重定向"]
```

### 1. Marker 到固定身份锚点

锚点由 SMPL-H 的 52 个关节和 61 个表面点组成，共 113 个。关节提供骨架结构，表面点辅助约束身体形状和局部扭转。锚点身份由人体模型定义，而非原始 marker 的 ID；初始化先将首帧 marker 规范化至身体中心坐标，再预测完整锚点。

### 2. 时序锚点跟踪

论文使用递归滑动窗口，默认窗口长 16 帧、相邻窗口重叠一半，并通过 768 维隐藏状态延续跨窗口信息。点 Transformer 建模每帧 marker 与锚点的关联，沿固定锚点身份的时间序列传播上下文，降低 raw marker 顺序不稳定带来的跨帧关联困难。长序列以窗口递推处理，内存保持有界。

### 3. 锚点到 SMPL-H：可微 Gauss–Newton

拟合变量包括窗口内逐帧身体姿态与全局平移，以及序列共享体型参数。加权目标由锚点观测残差、二阶时间平滑项和姿态先验组成。求解器学习每个时刻/锚点的观测置信度、平滑权重和先验权重，使其能按输入质量调整约束，而不把固定人工超参应用于所有标记配置。将损失从最终人体参数反传过迭代优化过程，实现可微拟合；自定义 CUDA 核计算 SMPL-H 运动学链 Jacobian。

## 工程实践与数据应用

| 环节 | 论文实现 | 使用/复现时注意 |
|---|---|---|
| 输入 | 无序 marker 点云；训练布局含 38–175 markers，并模拟丢失、重排、抖动、漂移、离群与 ghost points | 输入不需要 marker-to-body 身份标签，但输出仍依赖 SMPL-H 身体模型及锚点映射 |
| 时序建模 | 16 帧递归窗口，50% 重叠，768 维跨窗隐藏状态 | 以窗口处理任意长序列，离线/流式延迟和边界帧策略需按代码复核 |
| 求解 | 10 次可微 Gauss–Newton 迭代；CUDA 计算 Jacobian | 论文仓库尚无可运行代码，当前无法复核完整环境、模型权重或实际吞吐 |
| 下游数据 | 处理 134 段武术动捕，约 180 分钟、平均约 8,500 帧/段、22 个归一化风格标签，产出 SMPL-H 重建 | 项目页注明 HKMALA-Motion 数据 Coming soon；不可假定下载许可与源动捕相同 |

## 实验、评测与对比

### 合成多布局评测

CMU MoCap 与 GRAB 手部动作生成 2,068 段合成测试序列，按 80/20 划分训练/测试；marker 布局与噪声配置覆盖遮挡、离群和无序 marker。单一模型在多布局下报告约 8–10 mm MPJPE，并在大部分配置上优于布局专用或需要有序输入的基线；在有对应标签/布局要求的基线之间应区分公平比较条件。

### 真实扫描评测

在 SSM 同步人体扫描和 marker 上，对比 MoSh++。MoSh++ 获得更低的对称 Chamfer 距离（11.3 mm vs DirtyMoCap 15.3 mm），但它使用已知 marker-to-body 对应和 marker 专属表面偏置先验；DirtyMoCap 不使用这些先验，且有效 marker-to-mesh 距离更低（10.0 mm vs 7.5 mm）。两项指标回答不同问题，不应压成单一“整体胜出”结论。SSM 该布局没有手部 marker，因此手指细节主要由姿态先验与时间平滑补足。

### 求解速度与系统耗时

论文报告自定义 CUDA 求解器相对标准 PyTorch 实现最高 100× 加速；端到端分阶段时间仍约为初始化 3.08 ms/frame、轨迹预测 12.22 ms/frame、拟合 23.03 ms/frame，合计 38.33 ms/frame。因此 CUDA 的 solver 微基准不代表完整 pipeline 可按 30 FPS 实时运行。

## 与其他工作对比

- **SOMA / LocalMoCap / OpenMoCap：**这些对照依赖已知 marker 分类、布局或有序输入；DirtyMoCap 侧重单模型对多种布局泛化，主要优势是减少布局专用处理前置条件。
- **MoSh++：**基于已知对应关系做传统身体拟合；SSM 上扫描几何距离更优，而 DirtyMoCap 更贴近有效原始 marker 且不使用 marker 专属先验。
- **与机器人动作重定向的关系：**DirtyMoCap 输出人体 SMPL-H，并不直接产生机器人关节参考、接触可行性或机器人策略；它属于 [动作重定向流水线](../concepts/motion-retargeting-pipeline.md) 的上游清洗/参数化阶段，而非机器人重定向本身。

## 源码运行时序图

**不适用（截至 2026-10-14）：**官方 GitHub 仓库 README 仅写“Code will be released here.”，没有可运行的训练、推理或数据处理入口。因此当前不能据仓库画出真实运行时序，也不能独立复现论文所报速度。

## 局限与风险

- **未见布局会退化：**论文报告 held-out marker 配置直接推理误差上升；估计代理布局再微调可改善，但仍未达到使用精确配置微调的上限。
- **手部锚点不稳定：**手部顶点密集，投票映射可能不可靠；高精度时作者仍建议人工选择网格顶点。
- **SMPL-H 绑定较深：**迁移到其他参数化人体模型，需要模型专用锚点、残差与 Jacobian，并重新训练。
- **实时性尚不能由单项加速推出：**全流程论文测得约 38.33 ms/frame，且仓库尚不可运行，仍需在真实代码和硬件上确认吞吐与延迟。
- **数据与代码未发布：**项目页分别写明代码仓库入口但仓库仍待发布实现、HKMALA-Motion 数据 Coming soon；不能将 arXiv 摘要“code and data at project page”理解为当前已可下载/复现。

## 结论

**DirtyMoCap 的主要贡献是以稳定的人体代理锚点拆解“无序 marker 对应”问题，再把时序估计和身体模型拟合联合起来；当前证据支持跨配置重建能力，尚不支持即刻复现或端到端实时部署。**

1. **核心收益是减少 marker-layout 专用依赖**，而不是彻底消除人体模型、锚点设计或训练配置的先验。
2. **对比指标需分开读：**合成多布局 MPJPE 有竞争力；SSM 上 MoSh++ 的扫描 Chamfer 更低，DirtyMoCap 的有效 marker-to-mesh 距离更低。
3. **最高 100× 是 solver 相对 PyTorch 的加速声明；**论文 pipeline 分阶段总计约 38.33 ms/frame，不等价于 100× 实时全流程。
4. **下游价值在于先产出 SMPL-H 再交给重定向/跟踪链路；**它本身不是人形机器人策略或机器人动作重定向器。
5. **复现前先等代码/数据发布并查明许可，**尤其 HKMALA-Motion 的原始采集与重建文件使用边界。

## 关联页面

- [动作重定向流水线](../concepts/motion-retargeting-pipeline.md) — MoCap 清洗和 SMPL-H 标准化位于机器人重定向前端
- [人形参考运动与数据集对照](../comparisons/humanoid-reference-motion-datasets.md) — 区分光学动捕、人体参数化序列与已重定向机器人轨迹
- [MAMMA 多视角 markerless MoCap](./paper-mamma-markerless-motion-capture.md) — 多视角图像重建，与光学 marker 清洗路径互补
- [HumanTracker](./paper-humantracker.md) — 评价身体重建 tracker 的数据/指标工作流

## 参考来源

- [论文摘录](../../sources/papers/dirtymocap_arxiv_2609_19927.md)
- [官方项目页核查](../../sources/sites/dirtymocap.md)
- [官方代码仓库状态核查](../../sources/repos/dirtymocap.md)

## 推荐继续阅读

- [DirtyMoCap arXiv HTML](https://arxiv.org/html/2609.19927v1)
- [官方项目页（含方法图与 HKMALA-Motion 展示）](https://wanglongzju.github.io/DirtyMoCap-Project-Page/)
