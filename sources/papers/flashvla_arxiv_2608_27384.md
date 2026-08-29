# FlashVLA（流式异步动作解码）

> 来源归档（ingest）

- **标题：** FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.27384>
- **机构：** 加州大学圣地亚哥分校（UCSD）；麻省理工学院（MIT）
- **作者：** Zekai Li、Jiaming Tang、Zhijian Liu
- **项目页 / 博客：** <https://z-lab.ai/projects/flashvla/>
- **代码：** <https://github.com/z-lab/flashvla>
- **权重：** <https://huggingface.co/z-lab/flashvla-pi05-libero>（另有 `flashvla-pi05-robotwin`；集合 <https://huggingface.co/collections/z-lab/flashvla>）
- **入库日期：** 2026-08-29
- **一句话说明：** 对流匹配 VLA 做流式 chunk 解码：交错噪声缓冲 + chunk 级因果注意力，一次前向吐一块可执行动作；LIBERO 异步 2.43×、真机单卡 ≥30 Hz。

## 核心摘录（MVP）

### 1) 流匹配 VLA 的延迟与异步错配同源

- **摘录要点：** \(\pi_{0.5}\) 在 RTX 4090 上动作解码占逐步推理 **75%**（约 10 次串行去噪）。同步推理在 chunk 边界停机；异步推理用陈旧观测预测，lookahead 越大错配越大。根因是 **各 chunk 从纯噪声孤立解码**，既把全部去噪步压进一次调用，又让新 chunk 看不到在飞轨迹。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 问题设定。
  - [\(\pi_{0.5}\)](../../wiki/entities/paper-pi05-open-world-vla.md) — 基座策略。

### 2) 交错噪声缓冲 + chunk 级因果注意力

- **摘录要点：** 维护 \(N\) 个交错噪声水平的动作块；单次前向把全部块推进一级，稳态每步弹出一块干净动作（冷启动 \(N{-}1\) 步安全默认）。更噪的未来块只 attend 更干净的近执行块，隐式条件化在飞轨迹，**无需显式未来状态预测器**。建议 \(N\times C\) 接近预训练原生 chunk 长（\(\pi_{0.5}\) 为 50）。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 方法与消融。
  - [VLA](../../wiki/methods/vla.md) — 流匹配动作头加速。

### 3) 仿真与真机数字

- **摘录要点：** LIBERO \(d{=}1\)：成功率 **96.9%→97.8%**，逐步 **53.8→22.1 ms（2.43×）**；去掉因果掩码异步成功率约 **-10 pt**。RoboTwin 2.0 长程子集 **53.0%→89.6%（+36.6）**。RTX 5090 双视角推理 **20.3 ms**。Franka + RTX A4000：异步 \(d{=}2\) 维持 **30 Hz**，三任务均分 **80.0%→84.4%**，完成时间约 **1.3×**。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 评测读法。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 桌面操作语境。

### 4) 开源状态（截至 2026-08-29）

- **摘录要点：** **已开源** Apache-2.0。训练/评测/延迟基准可跑；LIBERO 与 RoboTwin 权重已发；真机部署脚本在仓内。
- **对 wiki 的映射：**
  - [flashvla 仓库](../repos/flashvla.md)
  - [z-lab FlashVLA 博客](../sites/z-lab-flashvla.md)

## 当前提炼状态

- [x] arXiv 摘要、方法与评测节已对齐摘录
- [x] 项目页/仓库/HF 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-flashvla.md` 新建
