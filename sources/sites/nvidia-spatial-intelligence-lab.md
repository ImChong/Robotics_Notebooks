# NVIDIA Spatial Intelligence Lab（SIL）

> 来源归档

- **标题：** NVIDIA Spatial Intelligence Lab
- **类型：** site / 研究实验室门户
- **URL：** <https://research.nvidia.com/labs/sil/>
- **机构：** NVIDIA Research
- **GitHub 组织：** <https://github.com/nv-tlabs>（公开描述为 *NVIDIA Spatial Intelligence Lab (SIL)*）
- **入库日期：** 2026-09-05
- **一句话说明：** NVIDIA 研究组 **Spatial Intelligence Lab**：推进 AI 在物理世界中的 **感知、建模与有意义交互** 基础技术；公开代码多挂在 **nv-tlabs**，项目页在 `research.nvidia.com/labs/sil/projects/*`。

## 实验室定位（门户 + GitHub 组织，2026-09-05）

> *Our goal is to advance foundational technologies enabling AI systems to perceive, model, and meaningfully interact with the physical world.*

- **成立：** GitHub 组织 `nv-tlabs` 创建于 **2019-04**；公开仓库约 **121** 个（API 核查）。
- **与 GEAR 分工：** [GEAR Lab](../../wiki/entities/nvidia-gear-lab.md) 偏 **通才具身智能体 / GR00T / SONIC**；SIL 更偏 **3D/4D 感知、神经重建、生成式世界模型、几何视频** 等 **空间智能** 基础（见本库已索引项目）。

## 本库已索引代表项目（非完整目录）

| 主题 | 项目页 / 代码 | 本库实体 |
|------|---------------|----------|
| 驾驶神经重建 | [instant-nurec](https://research.nvidia.com/labs/sil/projects/instant-nurec/) / [NVIDIA/instant-nurec](https://github.com/NVIDIA/instant-nurec) | [paper-instant-nurec](../../wiki/entities/paper-instant-nurec.md)、[nvidia-nurec](../../wiki/entities/nvidia-nurec.md) |
| 文生运动 | [kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) / [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | [kimodo](../../wiki/entities/kimodo.md) |
| 多智能体世界模型 | [gamma-world](https://research.nvidia.com/labs/sil/projects/gamma-world/) / [nv-tlabs/Gamma-World](https://github.com/nv-tlabs/Gamma-World) | [paper-gamma-world-multi-agent](../../wiki/entities/paper-gamma-world-multi-agent.md) |
| 技能嵌入 RL | [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) | [ASE 方法页](../../wiki/methods/ase.md) |
| 3D 生成 / 重建 | GET3D、3dgrut、Lyra、GEN3C、ViPE 等 | 见 [nv-tlabs 归档](../repos/nv_tlabs.md) |

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **部分开源** — 各项目独立仓库；门户本身无统一代码仓 |
| **代码入口** | 优先查 <https://github.com/nv-tlabs> 与各 `projects/*` 页 Footer 链接 |
| **许可** | **逐仓库**（MIT / Apache / NVIDIA Open Model License 等） |

## 命名消歧

- **SIL（本页）** = **Spatial Intelligence Lab**（研究组）。
- **SIL（工程）** = **Software-in-the-Loop**（[Isaac Sim 软件在环测试](../../wiki/concepts/software-in-the-loop.md)）——缩写相同、语义不同。

## 对 wiki 的映射

- 实体页：**`wiki/entities/nvidia-spatial-intelligence-lab.md`**
- GitHub 归档：[nv_tlabs.md](../repos/nv_tlabs.md)
- 工程 SIL 教程：[nvidia-isaac-sim-sil-tutorial.md](./nvidia-isaac-sim-sil-tutorial.md)
