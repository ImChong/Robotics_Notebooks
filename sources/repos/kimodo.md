# Kimodo — 原始资料归档

- **来源**：https://github.com/nv-tlabs/kimodo
- **类型**：repo
- **机构**：NVIDIA Research（nv-tlabs）
- **归档日期**：2026-05-14
- **GitHub**：约 2.3k stars，Apache-2.0（代码）；模型权重另见各 Hugging Face 页许可说明

## 一句话说明

**Kimodo**（**Ki**nematic **Mo**tion **D**iffusi**o**n）是在大规模（约 700 小时）商业友好型光学动捕数据上训练的 **运动扩散模型**，面向高质量 3D 人体与人形机器人运动生成；支持文本提示与多种运动学约束（全身关键帧、末端位姿、2D 路径与路点等）。

## 为什么值得保留

- NVIDIA 官方实现，配套 [技术报告 PDF](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf) 与 [在线文档](https://research.nvidia.com/labs/sil/projects/kimodo/docs)
- 与 **SOMA / Unitree G1 / SMPL-X** 多骨架变体及 **BONES-SEED** 等公开数据对齐，便于与 [ProtoMotions](../../wiki/entities/protomotions.md)、[MotionBricks](../../wiki/methods/motionbricks.md) 等人形栈对照
- 提供 **CLI、交互式时间线 Demo、评测基准**（[Kimodo-Motion-Gen-Benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark)）与 BONES-SEED 细粒度时间轴标注说明

## 能力与交付物（仓库自述要点）

| 模块 | 内容 |
|------|------|
| 推理 | `kimodo_gen` / `python -m kimodo.scripts.generate`；可选扩散步数、CFG 类型与权重 |
| 交互 Demo | `kimodo_demo`（本地 Gradio，默认 7860 端口） |
| 输出格式 | 默认 NPZ；G1 可写 MuJoCo qpos CSV；SMPL-X 可写 AMASS 兼容 npz（便于 GMR 等重定向） |
| 下游衔接 | README 说明与 ProtoMotions、MuJoCo 可视化、[GEAR-SONIC Web Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html) 的 Kimodo 文生运动集成 |

## 模型变体（摘要）

多版本按骨架与训练数据划分（SOMA / G1 / SMPL-X × Rigplay 1 / BONES-SEED 等），权重托管于 Hugging Face；默认推荐完整 Rigplay 1（约 700h）训练的变体以获得更强生成能力。

## 对 wiki 的映射

1. **[Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)**：人形尺度上的「扩散式轨迹生成 + 约束控制」工程范例
2. **[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)**：SMPL-X 输出与 GMR 等重定向管线的衔接说明
3. **[ProtoMotions](../../wiki/entities/protomotions.md)**：将 Kimodo 输出导入物理策略训练的官方推荐路径之一

## 引用（仓库 README）

```bibtex
@article{Kimodo2026,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and others},
  journal={arXiv:2603.15546},
  year={2026}
}
```
