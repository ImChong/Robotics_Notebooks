# Zero-to-CAD Qwen3-VL-2B 模型

> 来源归档（model / Hugging Face；核查日期：2026-10-23）

- **模型：** Zero-to-CAD-Qwen3-VL-2B
- **机构：** Autodesk Research
- **Hugging Face：** <https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B>
- **许可：** Apache-2.0（以模型卡标注为准）
- **论文：** <https://arxiv.org/abs/2604.24479>（[论文来源](../papers/zero_to_cad_arxiv_2604_24479.md)）
- **数据集：** <https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k>（[数据集归档](../datasets/zero-to-cad-100k.md)）
- **项目实体：** [Zero-to-CAD](../../wiki/entities/zero-to-cad.md)

## 模型卡核查

- 基于 **Qwen3-VL-2B-Instruct** 全量微调；输入为一个形状的 **8 张 256×256 多视角渲染图**，输出可执行的 **CadQuery Python** 建模程序。
- 模型卡说明训练数据来自完整 **Zero-to-CAD 1M** 合成集的 **979,633** 条样本；不能将它误写成仅用 100K 子集训练。
- 模型卡报告 Zero-to-CAD 测试集成功率 **82.1% / mean IoU 0.747**，ABC OOD 集成功率 **61.0% / mean IoU 0.377**；指标由发布方按 64³ voxel IoU、45° 旋转对齐协议计算。
- 官方卡片给出 Transformers 加载与推理示例，也展示用 CadQuery 将生成结果导出 STEP / STL。复杂装配、噪声或非 8 视角输入不在已验证范围内。

## 对 Robotics_Notebooks 的映射

- [Zero-to-CAD 实体页](../../wiki/entities/zero-to-cad.md)
- 相邻概念：[文字生成 CAD](../../wiki/concepts/text-to-cad.md)
