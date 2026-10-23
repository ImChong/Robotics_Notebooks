# NIDAR 官方实现与模型工件

> 来源归档

- **类型：** repo / model artifacts
- **代码：** <https://github.com/Zjj587/NIDAR>
- **模型：** <https://huggingface.co/zhangjunjie587/NIDAR>
- **项目页：** <https://nidar-web.github.io/>（[归档](../sites/nidar-project.md)）
- **论文：** <https://arxiv.org/abs/2609.36878>（[摘录](../papers/nidar_arxiv_2609_36878.md)）
- **代码许可证：** Apache-2.0；权重、校准文件、数据与第三方组件条款分别核对
- **核查日期：** 2026-10-23
- **沉淀到 wiki：** [NIDAR 实体页](../../wiki/entities/paper-nidar.md)
- **一句话说明：** 官方实现提供强度合成推理及 IRNet 训练资料，模型工件另在 HF 发布。

## 公开入口与边界

| 组件 | 公开内容 | 复现边界 |
|---|---|---|
| 推理 | GitHub inference/run_nidar_demo.py；提供 Waymo / nuScenes 输入准备说明 | 原始数据需从提供方获取，仓库不托管 |
| 模型 | HF 的 STN、U-Net + DeepLab IRNet checkpoint 与 Waymo 校准文件 | 模型卡列有文件校验信息；权重/校准许可须单独核对 |
| IRNet 训练 | GitHub 提供本地数据准备、训练/续训与评测指南 | HF 说明该配方不逐位重放托管 checkpoint；STN 重训不在模型卡公开范围 |
| 仿真集成 | 论文报告 UE5、Isaac Sim 与生成式 LiDAR 流程集成 | HF 模型卡注明 simulator demos 不包含在该模型包内；论文结果不等于实机验证 |
| 数据与许可 | 项目公开代码及部分模型工件 | 代码 Apache-2.0 不代表权重、数据、校准或第三方组件同许可 |

## 默认推理路线

```text
RGB → STN 伪 NIR → IRNet 反射率样分量 R
    → LiDAR 相机投影 → 数据集分位数重映射
    → 带强度样字段的现有点云
```

输出不是绝对回波光功率、传感器无关材料反射率或新 XYZ 几何。官方模型卡建议锁定下载 revision 并校验 SHA-256。

## 相关来源

- [NIDAR 项目页](../sites/nidar-project.md)
- [NIDAR 论文](../papers/nidar_arxiv_2609_36878.md)
