# Kimodo — 原始资料归档

- **来源**：https://github.com/nv-tlabs/kimodo
- **类型**：repo
- **机构**：NVIDIA Research（nv-tlabs）
- **归档日期**：2026-05-14（初版）；2026-05-21 深化（对齐项目页 / README / 论文）
- **项目页**：https://research.nvidia.com/labs/sil/projects/kimodo/
- **论文**：arXiv:2603.15546 — *Kimodo: Scaling Controllable Human Motion Generation*
- **GitHub**：约 2.4k stars，Apache-2.0（代码）；模型权重见各 Hugging Face 页许可（NVIDIA Open Model / R&D Model）

## 一句话说明

**Kimodo**（**Ki**nematic **Mo**tion **D**iffusi**o**n）是在大规模（约 **700 小时** [Bones Rigplay 1](https://bones.studio/datasets#rp01)）商业友好型光学动捕上训练的 **运动扩散模型**，面向高质量 3D 人体与人形机器人运动生成；支持文本提示与全身关键帧、末端位姿/旋转、2D 路点与稠密 2D 路径等 **运动学约束**。

## 为什么值得保留

- NVIDIA 官方实现，配套 [技术报告 PDF](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf)、[项目页](https://research.nvidia.com/labs/sil/projects/kimodo/) 与 [在线文档](https://research.nvidia.com/labs/sil/projects/kimodo/docs)
- 与 **SOMA（somaskel77）/ Unitree G1 / SMPL-X** 多骨架变体及 **BONES-SEED**、**Rigplay** 数据对齐，便于与 [ProtoMotions](../../wiki/entities/protomotions.md)、[MotionBricks](../../wiki/methods/motionbricks.md)、[GENMO/GEM](../../wiki/methods/genmo.md) 等人形栈对照
- 提供 **CLI、交互式时间线 Demo、公开评测基准**（[Kimodo-Motion-Gen-Benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark)）、评测管线代码与 [SEED-Timeline-Annotations](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) 细粒度时间轴文本标注

## 模型变体（README 摘要，2026-05）

| 模型 | 骨架 | 训练数据 | 备注 |
|------|------|----------|------|
| Kimodo-SOMA-RP-v1.1 | SOMA | Rigplay 1（700h） | 默认推荐；配合新 Benchmark |
| Kimodo-SOMA-SEED-v1.1 | SOMA | BONES-SEED（288h） | 便于与 SEED 训练的其他方法公平对比 |
| Kimodo-G1-RP-v1 | Unitree G1 | Rigplay 1 | 机器人演示数据 / MuJoCo CSV |
| Kimodo-SMPLX-RP-v1 | SMPL-X | Rigplay 1 | AMASS npz → [GMR](https://github.com/YanjieZe/GMR) 重定向 |

> v1.1 SOMA 模型主要为 Benchmark 兼容与小幅质量改进；权重在首次 `kimodo_gen` / `kimodo_demo` 时自动下载。

## 能力与交付物

| 模块 | 内容 |
|------|------|
| 推理 | `kimodo_gen` / `python -m kimodo.scripts.generate`；`--diffusion_steps`、`--cfg_type`（`nocfg` / `regular` / `separated`） |
| 交互 Demo | `kimodo_demo`（Gradio，127.0.0.1:7860）；时间线多 prompt + 多轨约束编辑 |
| 评测 | Benchmark 构造 + 生成 + 指标（质量 / 约束跟随 / 文本对齐）；嵌入 [TMR-SOMA-RP-v1](https://huggingface.co/nvidia/TMR-SOMA-RP-v1) |
| 输出 | 默认 NPZ（`posed_joints`、`smooth_root_pos`、`foot_contacts` 等）；G1 → MuJoCo qpos CSV；SMPL-X → AMASS npz |
| 下游 | ProtoMotions 导入、[MuJoCo 可视化脚本](https://github.com/nv-tlabs/kimodo)、[GEAR-SONIC Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html)、GMR 重定向 |

## 环境与限制（README）

- 全 GPU 推理约 **17GB VRAM**（文本编码占大头）；`TEXT_ENCODER_DEVICE=cpu` 可降至 <3GB VRAM、略增延迟
- 主要测试 GPU：RTX 3090 / 4090、A100；开发环境以 Linux 为主
- 2026-03-19 **Breaking**：模型 I/O 改用 SOMA **77-joint** 骨架（`somaskel77`）

## 对 wiki 的映射

1. **[Kimodo（实体页）](../../wiki/entities/kimodo.md)** — 架构、变体选型与下游管线
2. **[Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)** — 约束式全身扩散范例
3. **[ProtoMotions](../../wiki/entities/protomotions.md)** / **[SONIC](../../wiki/methods/sonic-motion-tracking.md)** — 生成轨迹 → 物理跟踪

## 关联原始资料

- [Kimodo 项目页](../sites/kimodo-project.md)
- [Kimodo 论文摘录](../papers/kimodo_arxiv_2603_15546.md)

## 引用（仓库 README）

```bibtex
@article{Kimodo2026,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and others},
  journal={arXiv:2603.15546},
  year={2026}
}
```
