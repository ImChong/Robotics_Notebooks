# Being-H0.7

> 来源归档（ingest）

- **标题：** Being-H0.7: A Latent World-Action Model from Egocentric Videos
- **类型：** paper
- **来源：** [项目页](https://research.beingbeyond.com/being-h07)；arXiv:2605.00078
- **入库日期：** 2026-05-12
- **最后更新：** 2026-05-12
- **一句话说明：** 用大规模第一人称人视频 + 机器人演示，在**潜空间**对齐「未来感知」监督，训练可部署的**世界–动作**先验；推理时不滚未来像素，直接输出动作。

## 核心论文摘录（MVP）

### 1) 问题动机：VLA 稀疏监督 vs 像素世界模型昂贵

- **链接：** <https://research.beingbeyond.com/being-h07>
- **摘录要点：** 标准 VLA 动作监督稀疏，易把视觉差异很大的状态塌缩成少数重复行为；像素级未来预测能利用稠密未来结构，但推理慢、脆弱，难塞进真实控制环。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md) — 与「端到端语言–视觉–动作」路线的数据效率与行为塌缩讨论互证
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 对比「视频即仿真」的高推理成本与本文的潜空间折中

### 2) 方法：可学习 latent queries + 打包双分支（先验 / 后验）

- **链接：** <https://arxiv.org/abs/2605.00078>
- **摘录要点：** 在「多模态理解」与「稠密动作生成」之间插入**可学习潜变量查询**，作为内部 world–action 工作空间。训练时**后验分支**可访问未来观测的紧凑嵌入，与仅含查询的**先验分支**在潜空间对齐 hidden；**单序列 + 共享上下文 + 双分支注意力掩码**，非两套独立网络。测试时只用指令、观测历史、本体状态与 latent queries，**不显式生成未来图像**。
- **对 wiki 的映射：**
  - [Latent Imagination](../../wiki/concepts/latent-imagination.md) — 同属「在紧凑表示里吸收未来结构」，但本文面向操作 VLA 式策略而非经典 RSSM rollout
  - [Being-H0.7 方法页](../../wiki/methods/being-h07.md) — 结构化归纳与部署栈

### 3) 数据规模、评测与系统部署

- **链接：** <https://research.beingbeyond.com/being-h07>
- **摘录要点：** 训练数据约 **20 万小时** 第一人称人视频 + **1.5 万小时** 机器人演示。仿真覆盖 LIBERO / LIBERO-plus、GR1、CALVIN、RoboCasa、Robotwin2 等；真机 12 任务、3 平台（PND Adam-U、Unitree G1、Franka FR3）、Linkerbot O6 手，对比 Being-H0.5、π₀.5、Fast-WAM。部署侧：**分块动作预测** + 客户端 **Universal Async Chunking (UAC)**；G1 上与 **AMO** 全身低层策略组合（上半身接口不变，AMO 管下肢与腰）。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md) — 与异步 chunk 缓冲控制延迟
  - [Imitation Learning](../../wiki/methods/imitation-learning.md) — 人视频 + 机器人演示的规模化模仿学习范式

## 关键术语

- **Latent world–action model**：用潜变量工作空间同时承担「世界相关预测结构」与「动作先验」接口。
- **Packed dual-branch attention**：先验槽位放 queries，后验槽位换为同形状未来嵌入，共享上下文、掩码隔离分支。
- **UAC (Universal Async Chunking)**：已提交动作前缀实时执行，异步请求下一 chunk，到达后再拼接后缀，平滑模型与网络抖动。

## 关联 Wiki 页面

- [Being-H0.7](../../wiki/methods/being-h07.md)
- [VLA](../../wiki/methods/vla.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)

## 当前提炼状态

- [x] 核心动机与潜空间接口
- [x] 双分支训练与部署假设
- [x] 数据规模、基准与系统栈（UAC / AMO）
- [ ] 细读 arXiv 全文后补充损失形式与架构细节
