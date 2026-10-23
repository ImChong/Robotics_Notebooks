# NIDAR: NIR-Guided Intrinsic Decomposition for Scalable Scene-Agnostic LiDAR Intensity Reconstruction（arXiv:2609.36878）

> 来源归档（ingest）

- **类型：** paper / lidar / intensity-synthesis / simulation / perception
- **论文：** <https://arxiv.org/abs/2609.36878>（2026-09-29；[PDF](https://arxiv.org/pdf/2609.36878)）
- **项目页：** <https://nidar-web.github.io/>（[核查记录](../sites/nidar-project.md)）
- **代码：** <https://github.com/Zjj587/NIDAR>（[核查记录](../repos/nidar.md)）
- **模型：** <https://huggingface.co/zhangjunjie587/NIDAR>
- **作者：** Junjie Zhang、Jie Yin、Kefei Qian、Jie Li、Mengpei Jia、Yajuan Dun、Wenbo Chu、Guofa Li
- **机构：** Chongqing University；Shanghai Jiao Tong University；Ministry of Industry and Information Technology；Western China Science City Innovation Center of Intelligent and Connected Vehicles（项目页作者脚注）
- **会议：** IROS 2026（arXiv 页面注明录用）
- **入库日期：** 2026-10-23
- **一句话说明：** 从 RGB 与仿真几何合成 LiDAR 强度样信号，通过伪 NIR、内禀分解和分布校准减少逐场景强度标注与拟合。

## 摘录 1：问题与方法

- 许多点云仿真忽略回波强度，或依赖真实强度标签和逐场景优化，增加数据采集与拟合成本。
- 论文提出前馈框架，从 RGB 外观和仿真几何合成稠密强度样观测；组件包括预训练伪 NIR 转换、层级内禀分解、几何调制和源域分布校准。
- 论文摘要称部署时无需目标场景强度标签或基于梯度的逐场景拟合。
- HF 模型卡说明默认推理路线为 `RGB → STN 伪 NIR → IRNet 的反射率样输出 R → LiDAR-camera 投影 → 分位数映射`；直接强度头 `I_hat` 不是最终默认输出。

**对 wiki 的映射：** 新建 [NIDAR 实体页](../../wiki/entities/paper-nidar.md)，关联 [LiDAR 传感](../../wiki/concepts/lidar-sensing.md) 和 [机器人仿真三层](../../wiki/concepts/robot-simulation-three-layers.md)。

## 摘录 2：评测

- 论文摘要称模型在 Waymo 数据上离线训练，固定权重与校准后在 Waymo 和 nuScenes 上评测。
- 与所比较的重建基线相比，论文报告有竞争力的像素精度与结构/感知保真度；具体数值需查论文表格及协议。
- 论文报告集成 Unreal Engine 5、Isaac Sim 与生成式 LiDAR 流程，并在两个仿真室内场景评测两个 intensity-aware SLAM 系统。
- 作者明确说明这些下游结果不构成实机验证；跨波段、相机配置、嵌入式部署及真实传感器验证仍待完成。

## 摘录 3：公开范围

- 官方 GitHub 提供代码；Hugging Face 模型卡列出 STN、IRNet checkpoint 和 Waymo 校准文件，并给出下载与推理说明。
- Waymo / nuScenes 原始数据由使用者按提供方流程获取，项目不托管副本。
- HF 模型卡说明公开 IRNet 配方并非托管 checkpoint 训练过程的逐位重放；STN 重训、训练数据及新训练权重不在该模型卡发布范围。
- 代码、权重、校准和第三方组件条款不同；代码许可证不能推出权重可商业使用。

## 一手资料

1. [arXiv 摘要与论文](https://arxiv.org/abs/2609.36878)
2. [官方项目页](https://nidar-web.github.io/)（资源链接仍显示 “Coming soon”，见核查记录）
3. [官方代码](https://github.com/Zjj587/NIDAR) 与 [HF 模型卡](https://huggingface.co/zhangjunjie587/NIDAR)
