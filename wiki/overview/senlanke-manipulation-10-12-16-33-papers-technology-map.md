---
type: overview
tags: [overview, survey, manipulation, vla, world-action-models, dexterous-manipulation, technology-map, wechat]
status: complete
updated: 2026-10-21
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
sources:
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "具身运控lab（森兰克周报）2026-10-12 至 10-16 Manipulation 盘点：33 篇论文按 10 个方向排列，33/33 独立详情节点（新建 19 / 复用 14）。"
---

# 操作方向前沿 33 篇阅读坐标：具身运控lab 10-12 至 10-16 期

> **本页定位**：为 [具身运控lab 周报 · 前沿论文动态（Manipulation 33 篇）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（公众号推送于 2026-10-20）提供按方向组织的阅读坐标。33 篇全部有独立的 `wiki/entities/paper-*.md` 详情节点，无重复节点。

## 一句话观点

**本期重心在 VLA 与世界动作模型（WAM）的结合：一边用潜在或紧凑的未来表示降低预测成本，另一边用测试时验证、后训练和恢复机制提升闭环可靠性；灵巧与接触方向则继续依赖分层、数据重用与可微仿真。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界动作模型，同时预测未来与动作 |
| WM | World Model | 世界模型，预测环境未来状态 |
| MPC | Model Predictive Control | 模型预测控制 |
| OPD | On-Policy Distillation | 在学生自身分布上的蒸馏 |
| JEPA | Joint-Embedding Predictive Architecture | 在潜在空间预测未来表征的架构 |

## 按方向阅读

### 一、VLA / 通用操作策略（5 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| FastVLA | 免训练的任务感知视觉 token 压缩，降低推理开销 | [paper-fastvla-task-adaptive-token-compression](../entities/paper-fastvla-task-adaptive-token-compression.md) |
| Multi-Teacher OPD | 多个冻结专家 Flow-VLA 蒸馏到一个学生 | [paper-multi-teacher-on-policy-distillation-flow-vla](../entities/paper-multi-teacher-on-policy-distillation-flow-vla.md) |
| ObjDream | 物体级理解与未来状态预测引导 VLA | [paper-objdream](../entities/paper-objdream.md)（复用） |
| VeriVLA | 测试时用潜在世界模型验证候选动作块 | [paper-verivla-test-time-verification](../entities/paper-verivla-test-time-verification.md) |
| ListenVLA | 音频作为 VLA 的条件输入 | [paper-listenvla-audio-conditioned-manipulation](../entities/paper-listenvla-audio-conditioned-manipulation.md) |

### 二、WAM / 世界模型与操作预测（6 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| T2-Z | 测试时优化环境潜变量，适应未知物理 | [paper-t2-z-environment-latents](../entities/paper-t2-z-environment-latents.md)（复用） |
| P³-WAM | 第一视角灵巧操作的统一感知、预测与规划 | [paper-p3-wam-egocentric-dexterous](../entities/paper-p3-wam-egocentric-dexterous.md) |
| UniW(A)M | 混合专家统一世界模型与 WAM 接口 | [paper-uniw-a-m-moe-world-world-action](../entities/paper-uniw-a-m-moe-world-world-action.md) |
| LoopWAM | 利用部署 rollout 做闭环策略改进 | [paper-loopwam](../entities/paper-loopwam.md)（复用） |
| WAVE-0 | 257M 潜在世界模型连接 VLA 与 WAM | [paper-wave-0-latent-future-prediction](../entities/paper-wave-0-latent-future-prediction.md) |
| Box-WAM | 用物体框轨迹替代稠密未来帧 | [paper-box-wam-box-trajectory-states](../entities/paper-box-wam-box-trajectory-states.md) |

### 三、在线 / 离线 RL 与后训练（3 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| VaM | 价值预测缓存为任务进度记忆，在线自主学习 | [paper-vam-value-as-memory](../entities/paper-vam-value-as-memory.md)（复用） |
| MERL | 想象器随策略演化，按可靠度控制想象数据 | [paper-merl-trust-calibrated-vla-post-training](../entities/paper-merl-trust-calibrated-vla-post-training.md) |
| HiRE-MoMa | 稀疏人工指导下的分层残差策略学习（移动操作） | [paper-hire-moma](../entities/paper-hire-moma.md)（复用） |

### 四、模仿学习与生成式动作策略（3 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| EgoJEPA | 从第一视角视频学潜在动作，免手工重定向 | [paper-egojepa-egocentric-latent-action](../entities/paper-egojepa-egocentric-latent-action.md) |
| ILA | 把交互潜变量作为接触操作的动作 | [paper-ila-interaction-latents-as-actions](../entities/paper-ila-interaction-latents-as-actions.md)（复用） |
| CFG-Policy | 无分类器引导统一扩散策略 | [paper-cfg-policy-unified-diffusion-policy](../entities/paper-cfg-policy-unified-diffusion-policy.md)（复用） |

### 五、双臂操作（2 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| Coordi-π | 左右手动作交错为独立 token | [paper-coordi-pi-bimanual-action-tokens](../entities/paper-coordi-pi-bimanual-action-tokens.md) |
| Bimanual RMA | 把潜在运动适应扩展到双臂 | [paper-bimanual-rma-latent-motor-adaptation](../entities/paper-bimanual-rma-latent-motor-adaptation.md) |

### 六、灵巧操作（4 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| DSP | 监督式自由交互 + 世界模型 MPC 学灵巧操作 | [paper-dsp-supervised-play](../entities/paper-dsp-supervised-play.md)（复用） |
| HiFun | 功能性灵巧操作的分层真机学习 | [paper-hifun](../entities/paper-hifun.md)（复用） |
| DexStitch | 从人类视频切分并拼接灵巧技能 | [paper-dexstitch-skill-stitching-human-videos](../entities/paper-dexstitch-skill-stitching-human-videos.md) |
| DexNinja | real2sim2real 的灵巧切割策略 | [paper-dexninja-real2sim2real-cutting](../entities/paper-dexninja-real2sim2real-cutting.md) |

### 七、柔性 / 可变形物体（2 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| ContinuousCloth | 物理信息神经 ODE 的连续时间布料动力学 | [paper-continuous-cloth-neural-ode](../entities/paper-continuous-cloth-neural-ode.md) |
| 接触自适应网格 | 仿真循环内按接触细化四面体网格 | [paper-contact-adaptive-meshing](../entities/paper-contact-adaptive-meshing.md) |

### 八、人形上半身操作（2 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| KinoGen | 运动学约束视频生成人形移动操作参考 | [paper-kinogen](../entities/paper-kinogen.md)（复用） |
| FACT（EgoForce） | 力–动作–轨迹的多模态基础模型 | [paper-fact-egoforce-human-dexterity](../entities/paper-fact-egoforce-human-dexterity.md)（复用；与 UCSD 的 FACT 失败感知训练是不同工作） |

### 九、长程任务、推理与规划（3 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| Mem4Act | 固定大小潜在记忆用于长程操作 | [paper-mem4act-latent-memory](../entities/paper-mem4act-latent-memory.md) |
| ForkCast | 反事实未来比较，决定 VLA 的恢复动作 | [paper-forkcast-counterfactual-recovery](../entities/paper-forkcast-counterfactual-recovery.md) |
| PILOT | 视觉子目标与策略先验引导世界模型规划 | [paper-pilot-visual-subgoal-world-model-planning](../entities/paper-pilot-visual-subgoal-world-model-planning.md)（复用） |

### 十、数据集、评测与仿真平台（3 篇）

| 论文 | 一句话 | 节点 |
|------|--------|------|
| LIBERO-PeRM | 个性化操作评测基准（312 个任务） | [paper-libero-perm-personalized-manipulation](../entities/paper-libero-perm-personalized-manipulation.md)（复用） |
| ForceBench | 机器人操作中的力信息融合评测 | [paper-forcebench](../entities/paper-forcebench.md)（复用） |
| CQDC-Warp | GPU 可微准动态仿真用于接触 MPC | [paper-cqdc-warp-differentiable-quasi-dynamic](../entities/paper-cqdc-warp-differentiable-quasi-dynamic.md) |

## 与其他期的关系

- 本期 VLA 与 WAM 方向与 [具身智能小站 10-18 期](embodied-station-2026-10-18-26-papers-technology-map.md) 有主题重叠，但论文不同，不合并节点。
- UniW(A)M（2610.18697）与 [UniWAM（2610.02054）](../entities/paper-uniwam-unified-world-action-model.md) 名字相近，是两篇不同工作。

## 参考来源

- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)
