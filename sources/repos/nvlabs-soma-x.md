# NVlabs / SOMA-X

> 来源归档（ingest）

- **标题：** SOMA — Unifying Parametric Human Body Models（SOMA-X 代码库）
- **类型：** repo + 官方文档站 + 技术报告（arXiv）
- **组织：** NVIDIA（NVlabs）
- **代码：** https://github.com/NVlabs/SOMA-X
- **文档站：** https://nvlabs.github.io/SOMA-X/stable/
- **PyPI：** https://pypi.org/project/py-soma-x/
- **技术报告：** https://arxiv.org/abs/2603.16858 — *SOMA: Unifying Parametric Human Body Models*
- **许可：** Apache-2.0（SMPL/SMPL-X 模型文件需单独许可）
- **入库日期：** 2026-06-17
- **一句话说明：** **SOMA** 提供 **统一拓扑与 rig** 作为 SMPL / SMPL-X / MHR / Anny / GarmentMeasurements 等异构参数化人体模型的 **canonical pivot**：在 **NVIDIA Warp** 上端到端可微、GPU 加速，使 **身份来源与姿态数据在推理时可自由组合**，并配套 **PoseInversion**（SMPL/MHR/AMASS→SOMA）、**soma.io**（NPZ/USD）与下游 **SEED / Retargeter / Kimodo / GEM** 生态。
- **沉淀到 wiki：** [SOMA-X](../../wiki/entities/soma-x.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) | SOMA 统一骨架是 NVIDIA 人形数据管线的 **表示层**；[SOMA Retargeter](../../wiki/entities/soma-retargeter.md) 在其上做 BVH→G1 |
| [GENMO / GEM](../../wiki/methods/genmo.md) | 视频人体估计与生成栈的 body model 互操作层 |
| [Kimodo](../../wiki/entities/kimodo.md) | 文本/约束运动扩散可在 **SOMA 骨架** 上输出 |
| [ProtoMotions](../../wiki/entities/protomotions.md) | 仿真训练接受 SOMA/SEED 格式参考 |
| [AMASS](../../wiki/entities/amass.md) | 提供 `convert_amass_to_soma` 批量 SMPL→SOMA 工具 |

---

## README / 文档站归纳

1. **核心问题：** SMPL、SMPL-X、MHR、Anny、GarmentMeasurements 等 **网格拓扑、关节层级、参数化互不兼容**，跨模型组合优势需为每对模型写 **bespoke adapter**。
2. **SOMA 解法：** **不替换** 现有模型，而是把多样 **rest shape** 映射到 **单一共享表示**，任意支持的身份模型可用 **统一动画管线** 驱动；推理时可 **mix-and-match** 身份与姿态。
3. **实现：** `SOMALayer` 全身体模型；`PoseInversion.fit()` 逆姿态拟合（**Analytical** 逆 LBS + Newton-Schulz，默认 ~1200 FPS；可选 **Autograd FK** 精修）；`soma.geometry`（FK、LBS、skeleton fitting、Warp kernels）；`soma.io`（NPZ、USD）。
4. **身份模型（5 类）：** MHR（默认）、Anny（儿童）、SMPL/SMPL-X、SOMA-shape（128 PCA）、GarmentMeasurements（CAESAR 服装测量）。
5. **统一 Pose Correctives（Beta）：** 为原本无 correctives 的模型（如 Anny、GarmentMeasurements）也提供姿态相关形变修正。
6. **转换工具：** `smpl2soma`、`mhr2soma`、`convert_amass_to_soma`；输出 NPZ 含 `poses`、`root_translation`、`joint_names`、`per_vertex_error` 等。
7. **安装：** `pip install py-soma-x`；可选 `[smpl]`、`[anny]`；资产首次从 HuggingFace 自动下载；开发克隆需 **Git LFS**。

---

## 生态互链（README Related Work）

| 项目 | 角色 |
|------|------|
| [GEM-X](https://github.com/NVlabs/GEM-X) | SOMA 视频姿态估计 |
| [Kimodo](https://github.com/nv-tlabs/kimodo) | SOMA 可控文生运动 |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | 大规模 SOMA 格式动捕 + G1 重定向数据 |
| [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) | SOMA→G1 |
| [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | 物理仿真与人形学习 |
| [GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | 人形行为基础模型（README 标注 coming soon） |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/soma-x.md`**：统一人体表示实体页 + **身份→SOMA→下游** Mermaid + 与 SMPL/重定向生态对照。
- 交叉更新：`wiki/entities/soma-retargeter.md`、`wiki/concepts/motion-retargeting.md`、`wiki/methods/genmo.md`、`wiki/entities/kimodo.md`、`references/repos/retarget-tools.md`。

---

## 外部参考

- Saito et al., *SOMA: Unifying Parametric Human Body Models*, [arXiv:2603.16858](https://arxiv.org/abs/2603.16858)
- [NVlabs/SOMA-X（GitHub）](https://github.com/NVlabs/SOMA-X)
- [SOMA-X 文档站](https://nvlabs.github.io/SOMA-X/stable/)
