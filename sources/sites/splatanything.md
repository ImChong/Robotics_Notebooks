# SplatAnything / InfiniSplat v3 项目页

> 来源归档（ingest · 项目主页）

- **标题：** SplatAnything (InfiniSplat v3): Gaussian Spacing Normalization for Feed-Forward 3DGS across Arbitrary Scenes
- **类型：** site / project-page
- **官方入口：** <https://zju3dv.github.io/SplatAnything/>
- **论文：** <https://arxiv.org/abs/2611.08264>，归档见 [论文](../papers/splatanything_arxiv_2611_08264.md)
- **代码：** <https://github.com/zju3dv/SplatAnything>，归档见 [代码](../repos/splatanything.md)
- **模型权重：** <https://huggingface.co/ritianyu/SplatAnything>
- **Demo：** <https://huggingface.co/spaces/ritianyu/SplatAnything>
- **入库日期：** 2026-11-10
- **核对范围：** 官网标题、方法架构图、尺度归一化对比、应用展示、外链资源、官方仓 README。
- **通读：** 已完成
- **开源状态：** 已公开推理代码和模型权重链接；官方源码为推理版，未包含论文完整训练管线。代码/模型的使用受官方许可证约束，不能默认适用宽松商业许可。
- **一句话说明：** 跨物体、室内、街景、航拍的前向 3D Gaussian 重建，支持多相机 Rig-Recon 与全景 Pano-Recon。

## 项目页核对结论

- 项目自称 InfiniSplat 系列 v3，以任意场景、相机布局、稀疏输入的泛化为核心卖点。
- 主页可见方法图：冻结几何预测 → Gaussian Spacing Normalization → 可训练 Gaussian backbone 与 GS head → 3DGS。
- 官方 GitHub README 给出 `inference.py` 的 `--mode rgb`（模型预测相机）和 `--mode pose`（读取 `prior_camera.json`）两条路径，以及 3DGS PLY 与 RGB/深度导出。
- `Rig-Recon`（标定多相机）和 `Pano-Recon`（全景图转多视角）是源码 README 描述的应用；不是在真实机器人闭环里已验证的物理仿真。
- 主页 Video 区显示「Demo video coming soon」；不要将三维交互演示误认成公开训练系统。

## 对 wiki 的映射

- [SplatAnything 论文/项目统一实体](../../wiki/entities/paper-splatanything.md)
- [论文原文归档](../papers/splatanything_arxiv_2611_08264.md)
- [推理源码归档](../repos/splatanything.md)
