---
type: entity
project_id: mdm-human-motion-diffusion
tags: [paper, human-motion, motion-generation, generative-motion, humanoid]
status: draft
updated: 2026-11-11
arxiv: "2209.14916"
project: https://guytevet.github.io/mdm-page/
code: https://github.com/GuyTevet/motion-diffusion-model
original:
  - ../../sources/papers/mdm_human_motion_diffusion_primary.md
sources:
  - ../../sources/papers/mdm_human_motion_diffusion_primary.md
  - ../../sources/sites/mdm-human-motion-diffusion-primary.md
  - ../../sources/repos/mdm-human-motion-diffusion-official.md
related:
  - ../comparisons/generative-motion-to-humanoid-closed-loop.md
  - ../methods/diffusion-motion-generation.md
  - ./ardy.md
summary: "Transformer 扩散模型直接预测去噪动作样本，允许位置、速度与足接触几何损失；可做文生动作、动作类条件生成及编辑。"
---

# MDM：文本与动作条件人体运动扩散

## 一句话定义

Transformer 扩散模型直接预测去噪动作样本，允许位置、速度与足接触几何损失；可做文生动作、动作类条件生成及编辑。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| MDM | Motion Diffusion Model | 人体动作扩散模型 |
| CFG | Classifier-Free Guidance | 条件与无条件引导 |
| CLIP | Contrastive Language–Image Pre-training | 文本编码器 |

## 为什么重要

- 将本研究放进从可控角色动画、实时生成到 G1 真机视觉闭环的对照坐标；输入和输出属于不同技术层级。
- 对工程部署而言，必须分别核查动作生成时间、动作跟踪控制频率和整机成功率。

## 核心信息

| 项 | 内容 |
|---|---|
| 论文/会议 | [ICLR 2023](https://arxiv.org/abs/2209.14916) |
| 机构 | 特拉维夫大学（Tel Aviv University） |
| 作者 | Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano |
| 项目主页 | [作者官方入口](https://guytevet.github.io/mdm-page/) |
| 开源 | [官方仓库](https://github.com/GuyTevet/motion-diffusion-model)（依赖和模型文件另行准备） |
| 原文阅读状态 | **未完成全文与全部附录核对，暂为 draft** |

## 流程总览

基于可核对的一手论文机制绘制主干模块关系：

```mermaid
flowchart LR
  A["文本或动作标签"] --> B["条件编码"]
  C["加噪运动序列"] --> D["Transformer 去噪器"]
  B --> D
  D --> E["预测干净动作 x0"]
  E --> F["几何及足接触损失 / 采样"]
  F --> G["人体运动帧序列"]
```

## 核心原理

采用 classifier-free 条件生成与 Transformer 去噪；预测原始动作样本而非直接噪声，使关节位置、速度和足接触相关损失可参与训练。它主要生成给定时长的动作序列。

## 实验与评测

原文在 HumanML3D、KIT-ML 等文本动作基准与动作类别条件任务上评测；官方仓库特别提示早期评价代码曾修复，不应跨版本直接抄旧表。

## 结论

**研究价值在于 Transformer 扩散模型直接预测去噪动作样本，允许位置、速度与足接触几何损失；对比时必须看清控制层级和数据口径。**

1. 生成模型输出的运动序列并不自动满足真实机器人动态平衡和接触约束。
2. 相关数值只对应论文实验配置，不能横向等同于另一系统的真机成功率。
3. 将该工作与相邻项目区分为 **生成、跟踪、视觉条件适配、闭环执行** 等不同接口。

## 源码运行时序图

依据官方 README 及已核对的代码入口，仅描述可明确确认的运行链路：

```mermaid
sequenceDiagram
  autonumber
  participant User as CLI
  participant Data as get_dataset_loader
  participant Model as create_model_and_diffusion
  participant Sampler as p_sample_loop
  participant Viz as recover_from_ric / plot_3d_motion
  User->>Data: python -m sample.generate + 条件
  User->>Model: 构建并加载 checkpoint
  Data-->>Sampler: 运动维度与条件
  Model-->>Sampler: 模型及 diffusion
  Sampler-->>Viz: 生成的动作张量
  Viz-->>User: results.npy 与渲染视频
```

## 工程实践

官方仓库含 `train.train_mdm`、`sample.generate`、`eval.eval_humanml`；公开预训练权重与 50-step 加速版本（与原论文训练设定应分开对照）。

## 局限与风险

文生运动学序列并不包含真实机器人的接触估计和控制反馈；原始 MDM 的单次整段生成不等价于 0.5 秒视觉闭环。

**阅读阻塞：** 仅核对论文的主要方法/结果段落、作者项目页及可用仓库入口；尚未完成全部原文与附录的通读及代码运行验证。按知识库规则保留 draft，未声称原文通读。

## 关联页面

- [从动作生成到人形闭环：方法与证据对比](../comparisons/generative-motion-to-humanoid-closed-loop.md)
- [扩散式动作生成](../methods/diffusion-motion-generation.md)
- [ARDY：交互可控动作生成](./ardy.md)
- [Kimodo：运动生成主干](./kimodo.md)

## 参考来源

- [mdm_human_motion_diffusion_primary.md](../../sources/papers/mdm_human_motion_diffusion_primary.md)
- [mdm-human-motion-diffusion-primary.md](../../sources/sites/mdm-human-motion-diffusion-primary.md)
- [mdm-human-motion-diffusion-official.md](../../sources/repos/mdm-human-motion-diffusion-official.md)

## 推荐继续阅读

- [原始论文](https://arxiv.org/abs/2209.14916)
- [作者项目主页](https://guytevet.github.io/mdm-page/)
- [官方源码](https://github.com/GuyTevet/motion-diffusion-model)
