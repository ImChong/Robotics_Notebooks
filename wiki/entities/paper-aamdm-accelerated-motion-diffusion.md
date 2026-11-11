---
type: entity
project_id: aamdm-cvpr-2024
tags: [paper, human-motion, motion-generation, generative-motion, humanoid]
status: complete
updated: 2026-11-11
arxiv: "2401.06146"
project: https://easypapersniper.github.io/projects/AAMDM/AAMDM.html
original:
  - ../../sources/papers/aamdm_accelerated_motion_diffusion_primary.md
sources:
  - ../../sources/papers/aamdm_accelerated_motion_diffusion_primary.md
  - ../../sources/sites/aamdm-accelerated-motion-diffusion-primary.md
related:
  - ../comparisons/generative-motion-to-humanoid-closed-loop.md
  - ../methods/diffusion-motion-generation.md
  - ./ardy.md
summary: "用潜空间 DD-GAN 快速生成动作草稿，再由自回归扩散模块少量步数精修以实现交互速度；独立于 A-MDM。"
---

# AAMDM：快速生成与扩散精修双模块

## 一句话定义

用潜空间 DD-GAN 快速生成动作草稿，再由自回归扩散模块少量步数精修以实现交互速度；独立于 A-MDM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| AAMDM | Accelerated Auto-regressive Motion Diffusion Model | 本项目缩写 |
| DD-GAN | Denoising Diffusion GAN | 快速生成模块 |
| ADM | Auto-regressive Diffusion Model | 精修模块 |

## 为什么重要

- 将本研究放进从可控角色动画、实时生成到 G1 真机视觉闭环的对照坐标；输入和输出属于不同技术层级。
- 对工程部署而言，必须分别核查动作生成时间、动作跟踪控制频率和整机成功率。

## 核心信息

| 项 | 内容 |
|---|---|
| 论文/会议 | [CVPR 2024](https://arxiv.org/abs/2401.06146) |
| 机构 | 佐治亚理工学院（Georgia Tech）；不列颠哥伦比亚大学（UBC）；北京航空航天大学（Beihang University）；西蒙菲莎大学（Simon Fraser University） |
| 作者 | Tianyu Li, Calvin Qiao, Guanqiao Ren, KangKang Yin, Sehoon Ha |
| 项目主页 | [作者官方入口](https://easypapersniper.github.io/projects/AAMDM/AAMDM.html) |
| 开源 | 项目页未核实到官方可执行源码 |
| 原文阅读状态 | **已完成主要论文全文、适用附录及原文结果复核**；源码仅核对主要运行路径 |

## 流程总览

基于可核对的一手论文机制绘制主干模块关系：

```mermaid
flowchart LR
  A["历史运动与当前用户命令"] --> B["编码到低维动作空间"]
  B --> C["DD-GAN Generation: 3 steps"]
  C --> D["ADM Polishing: 2 steps"]
  D --> E["潜空间动作输出"]
  E --> F["解码人体姿态"]
  F --> G["下一次自回归条件"]
```

## 核心原理

首先学习人体动作的潜空间表示；DD-GAN 使用少步逆扩散快速产生候选，ADM 用额外扩散步骤修正细节。论文描述配置为生成三步、精修两步；不是与 A-MDM 相同的模型。

## 实验与评测

CVPR 2024 论文在 LaFAN1、交互控制及人造多模态数据上做定量与消融；需区分在线虚拟角色动作与硬件控制延迟。

## 原文复核摘录（正文证据）
- **生成序列：** Autoencoder 把 338 维 pose 与未来路线特征映射到 64 维 embedding（52 learned latent + 12 规划特征）；DD-GAN 生成 3 步 + ADM 精修 2 步，共 5 步，输出下一帧后迭代。
- **实验：** LaFAN1 25 段、约 100k 帧转换、26.67 分钟；表1 AAMDM 173 FPS、AMDM200 4.72 FPS，方向控制误差 TE-UC=0.034；表2 无 polishing 出现长序列失稳，去掉 embedding FID 显著上升。
- **附录：** §6 的 Algorithm 1 明确先训练/滚动精修 ADM，再通过 DD-GAN G/D 优化快速生成分支；训练窗口 10，batch 64，ADM LR 3e-4，G/D LR 1e-4。
- **参考：** §3.1–§3.6、§4.1–§4.3、§5、§6；[论文原文](https://arxiv.org/html/2401.06146)。

## 结论

**研究价值在于 用潜空间 DD-GAN 快速生成动作草稿，再由自回归扩散模块少量步数精修以实现交互速度；对比时必须看清控制层级和数据口径。**

1. 生成模型输出的运动序列并不自动满足真实机器人动态平衡和接触约束。
2. 相关数值只对应论文实验配置，不能横向等同于另一系统的真机成功率。
3. 将该工作与相邻项目区分为 **生成、跟踪、视觉条件适配、闭环执行** 等不同接口。

## 源码运行时序图

**不适用：** 作者一手项目页仅提供论文与视频，本次未确认官方可运行实现。

## 工程实践

官方作者项目页公开论文和演示视频；截至本次核查没有在该项目页找到可确认的官方可运行源码仓库。

## 局限与风险

尚无已确认官方源码入口，不能绘制虚构调用时序；论文实验不支持推断 G1 真机运行。

**核查边界：** 已复核论文正文、结果与可用附录；代码仓库仅核对 README 与关键运行入口，未运行真机部署或完整训练。论文内已发布结果不等同于独立复现。

## 关联页面

- [从动作生成到人形闭环：方法与证据对比](../comparisons/generative-motion-to-humanoid-closed-loop.md)
- [扩散式动作生成](../methods/diffusion-motion-generation.md)
- [ARDY：交互可控动作生成](./ardy.md)
- [Kimodo：运动生成主干](./kimodo.md)

## 参考来源

- [aamdm_accelerated_motion_diffusion_primary.md](../../sources/papers/aamdm_accelerated_motion_diffusion_primary.md)
- [aamdm-accelerated-motion-diffusion-primary.md](../../sources/sites/aamdm-accelerated-motion-diffusion-primary.md)

## 推荐继续阅读

- [原始论文](https://arxiv.org/abs/2401.06146)
- [作者项目主页](https://easypapersniper.github.io/projects/AAMDM/AAMDM.html)
