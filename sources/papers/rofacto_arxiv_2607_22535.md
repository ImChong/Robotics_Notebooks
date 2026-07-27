# Robot-Factored World Models via Robot Rendering

> 来源归档（ingest）

- **标题：** Robot-Factored World Models via Robot Rendering
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.22535>
  - <https://ar5iv.labs.arxiv.org/html/2607.22535>
  - <https://bjkim95.github.io/rofacto/>
  - <https://github.com/bjkim95/rofacto>（项目页 Code 按钮；截至入库日 API 404）
- **作者：** Byungjun Kim, Taeksoo Kim, Hyunsoo Cha, Hanbyul Joo
- **机构：** 首尔大学（Seoul National University）；RLWRLD（Hanbyul Joo 第二单位）
- **入库日期：** 2026-07-27
- **一句话说明：** 把 **动作实现（controller+kinematics → nominal trajectory）** 与 **URDF 机器人渲染** 从视频世界模型中 **外提**：模型只看到相机对齐的网格 RGB + 末端/场景深度，学习场景如何响应；在 DROID / RoboCasa-GR1 上优于向量/位姿条件基线，并支持未见 embodiment 与人手演示重定向。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://bjkim95.github.io/rofacto/> — 方法/结果/消融/BibTeX；头部 **Code** 链到 GitHub |
| GitHub | <https://github.com/bjkim95/rofacto> — **404 Not Found**（截至入库日） |
| 可运行代码 / 权重 | **否** |
| 结论 | **宣称将开源 / 项目页已列 URL 但仓未公开** |

## 核心论文摘录（MVP）

### 1) 问题：动作信号夹在「命令」与「已交互状态」之间

- **链接：** <https://arxiv.org/abs/2607.22535> §1–§3.2
- **摘录要点：** 直接条件化 raw action 逼模型同时学 **机器人局部实现** 与 **场景响应**；条件化 logged realized state 则 **泄漏** 接触/柔顺/闭环修正。中间信号是 **nominal trajectory**：经本机控制器与运动学 rollout、**场景交互前** 的期望机器人运动，部署时可得。
- **对 wiki 的映射：**
  - [Rofacto（论文实体）](../../wiki/entities/paper-rofacto.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)
  - [DWM（Dexterous World Models）](../../wiki/methods/dwm.md) — 同实验室视觉条件路线对照

### 2) 方法：URDF 渲染接口 + 深度消歧

- **链接：** arXiv §3
- **摘录要点：**
  - \(\bm{q}=\Phi_R(\bm{a};\bm{q}_0)\) → \(\Pi_R\) 渲染 mesh RGB + EEF depth；静态场景经 \(\Pi_S\) 得背景 RGB + scene depth（固定机位可重复首帧）。
  - 条件 \(p(V\mid B^{\mathrm{rgb}}, D^{\mathrm{scene}}, M^{\mathrm{rgb}}, D^{\mathrm{eef}}, \mathcal{T})\)；文本 \(\mathcal{T}\) **仅场景上下文**，不含动作/结果描述。
  - 骨干：Wan2.1-Fun InP 视频修复式残差动力学（对齐 DWM）；VAE 编码四路条件后拼接；flow-matching 训练。
- **对 wiki 的映射：**
  - [Rofacto](../../wiki/entities/paper-rofacto.md)
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 向量/位姿条件对照

### 3) 主实验与消融

- **链接：** arXiv §4；项目页
- **摘录要点：**
  - 数据：DROID enhanced-extrinsic ~41.6k clips；RoboCasa-GR1（DiT4DiT 执行）~9.4k；81 帧 / 480×832 / 16 fps。
  - Wan 14B：Rendered+depth 相对 AdaLN state vector — DROID PSNR **18.57→21.87**；RoboCasa **17.67→24.61**。
  - SVD：Rendered mesh 优于 Ctrl-World 式 pose（PSNR **23.15→25.05**）。
  - 消融：Raw-action mesh → Nominal → +depth，逐步提升；零样本 xArm+Inspire、双臂合成；DexYCB 人手→机器人视频。
- **对 wiki 的映射：**
  - [Rofacto](../../wiki/entities/paper-rofacto.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 4) 局限

- **链接：** arXiv §5
- **摘录要点：** 需已知 URDF 与相机–机器人标定；动相机时静态上下文依赖可用场景表示；DROID 偏成功轨迹，失败接触样本少。**代码截至入库日未公开。**
- **对 wiki 的映射：**
  - [Rofacto](../../wiki/entities/paper-rofacto.md)
