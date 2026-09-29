# OCEC（Open Closed Eyes Classification）

> 来源归档

- **标题：** OCEC — Open Closed Eyes Classification
- **类型：** repo
- **维护者：** [PINTO0309](https://github.com/PINTO0309)
- **代码：** <https://github.com/PINTO0309/OCEC>
- **PINTO Model Zoo：** <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/476_OCEC>
- **ONNX 权重：** <https://github.com/PINTO0309/OCEC/releases>（`ocec_p` … `ocec_l`）
- **Zenodo：** <https://doi.org/10.5281/zenodo.17505461>
- **参考数据集：** [MichalMlodawski/closed-open-eyes](https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes)（见 [closed-open-eyes 归档](../datasets/closed-open-eyes.md)）
- **许可：** MIT（GitHub License badge）
- **入库日期：** 2026-09-29
- **一句话说明：** 面向 **极小眼部 ROI（默认 24×40）** 的开/闭眼二分类：提供 P–L 六档 ONNX（112 KB–6.4 MB，CPU 亚毫秒级）、完整数据管线（WholeBody34 眼框裁剪 → parquet → 训练 → 导出）与 `demo_ocec.py` 联调 **人体检测 + OCEC** 的实时眨眼/ wink 估计。
- **步骤 2.5（开源核查）：** **已开源**（2026-09 GitHub 复核）— 训练/推理/数据脚本、`uv` 环境与 Releases 上 ONNX 均可公开获取；无单独项目页，以仓库 README 与 Zenodo DOI 为准。

## 为何值得保留

- **边缘感知样板：** 在真实场景中眼部 bbox 常仅 **约 20×11 px** 量级；OCEC 把输入分辨率钉在 **24×40**，避免对大图做无效高分辨率分类，适合机载/工控机 **低延迟** 通道。
- **与 PINTO 生态衔接：** Model Zoo **476_OCEC** 与 Releases 提供即下即用的 ONNX；demo 默认搭配 **DEIMv2 + DINOv3 WholeBody34** 眼检测 ONNX（需自备 detector 权重，README 有示例文件名）。
- **机器人相关读法：** 可用于 **操作者状态监测**（疲劳/确认眨眼）、**XR/遥操作** 侧通道、或 **人机共存** 场景下的轻量视觉 cue——非 Loco/Manip 主栈，但是 **ONNX 微模型 + 检测级联** 的可复现参考。

## 对 wiki 的映射

- [OCEC 实体](../../wiki/entities/ocec.md) — wiki 主节点
- [ONNX](../../wiki/entities/onnx.md) — 导出与机载推理契约
- [ONNX Runtime vs MNN vs TensorRT](../../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md) — 选型 `demo_ocec.py -ep cuda|tensorrt|cpu`
- [closed-open-eyes 数据集](../datasets/closed-open-eyes.md) — HF 参考数据与 OCEC 自建 parquet 的对照
