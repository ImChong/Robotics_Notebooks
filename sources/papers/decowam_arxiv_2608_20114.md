# DECOWAM（arXiv:2608.20114）

> 来源归档（ingest）

- **标题：** DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation
- **类型：** paper / world-action-model / legged-mobile-manipulation
- **arXiv：** <https://arxiv.org/abs/2608.20114>
- **机构：** 清华大学；上海人工智能实验室；哈尔滨工业大学；云深处科技（DEEP Robotics / 杭州云深处）
- **入库日期：** 2026-08-22
- **一句话说明：** 腿足移动操作 **全身 WAM**：在冻结适配后的 **FastWAM** 上，用残差 adapter、未来瓶颈、基座/臂对抗解耦 latent 与基座速度 ego-motion 条件，联合预测未来 RGB 与 14-D 全身 action chunk；配套 **ARMDOG** 真机数据集。

## 开源状态（步骤 2.5，2026-08-22）

| 资源 | 状态 |
|------|------|
| arXiv HTML/PDF | **已发布** |
| 项目页 / GitHub / 数据集 | **截至入库日未列公开链接** — 确认未开源 |

## 核心论文摘录

### 1) 问题与因子分解（Abstract / §I）

- **核心贡献：** 固定基座 VLA/WAM 不显式区分 **相机 ego-motion**、**基座速度** 与 **臂操作**；腿足移动操作相机随基座运动，像素位移混合场景动力学与自运动。DECOWAM 在 **FastWAM（Wan-2.2 + ActionDiT）** 上引入四类可训练接口：残差 adapter、**action-equivalent future bottleneck**（特权未来 latent 蒸馏）、**GRL 分离的 base/arm latent**、**base-velocity token 条件化视频分支**。
- **对 wiki 的映射：**
  - [DECOWAM 论文实体](../../wiki/entities/paper-decowam.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) ARMDOG 数据集（§I / Dataset）

- **核心贡献：** 四足+6-DoF 臂平台；**217 episodes**、27 任务文件夹、**56,041** 同步帧；15 Hz RGB + \(T\times14\) 全身 state/action（臂/夹爪/基座速度/padding）+ 语言与预计算嵌入。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)（腿足移动操作 / 四足+臂）

### 3) 两阶段参数高效适配（§IV-B）

- **Stage 1：** 全参数 FastWAM 对齐 ARMDOG（50k steps）。
- **Stage 2：** 冻结 \(\Theta^{(1)}\)，仅训 \(\Phi=\{\phi_{\mathrm{adp}},\phi_q,\phi_{\mathrm{ba}},\phi_{\mathrm{ego}}\}\) — **25.95M** 可训练参数（相对全量 6020.75M 约 **232×** 缩减）。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)（FastWAM 系）

### 4) 评测要点（§VI）

- **开环 replay：** 相对 FastWAM，action MSE **−21.7%**；F-MSE / A-MSE 同步改善。
- **真机闭环：** 每法 **79 trials**；**全身协调（WBCM-SR）** 与 **基座位移鲁棒性** 领先；任务完成率与最强基线相当。
- **部署：** 因果推理仅 \((x_0,s_0,\ell)\)；evaluator 延迟约 **+11.4%** vs FastWAM。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md)（48-step chunk）

### 5) 局限

- ARMDOG 规模仍有限；未开源复现栈；与轮式/固定基座 WAM 的直接迁移未验证。
