# LeTools-Learning：Roban 全身 VLA 官方教程

> 来源归档（官方文档）

- **教程：** <https://www.letools.lejurobot.com/docs.html#tutorials/roban_wholebody_vla.md>
- **配套代码：** <https://github.com/LejuRobotics/LeTools-Learning>
- **相关控制器仓：** <https://gitcode.com/OpenLET/LejuLab-Deploy>
- **入库日期：** 2026-10-11
- **一句话说明：** 面向 Roban 机型的全身 VLA 操作指南，覆盖 rosbag→LeRobot v3、OpenPI / GR00T N1.7 / LingBot-VLA v2 训练及 CycloneDDS + HEFT 真机推理。

## 教程流程与规格

- **数据准备：** `RobanRosbag2Lerobot.yaml` 指定 `platform_type: roban`，运行 `kuavo_data/roban_data/CvtRosbag2Lerobot.py`，输出 `observation.state` 23 维、`action` 32 维、头部相机图像的视频数据。
- **时序：** 训练与 HEFT 控制按 50 Hz；示例输入头部相机。字段定义由 Roban wholebody modality 配置约束。
- **模型：** 教程展示 OpenPI、Isaac GR00T N1.7、LingBot-VLA v2 三条独立训练配置；对应全身 action chunk 分别按模型配置处理。
- **部署：** 边侧机运行模型 server，通过 CycloneDDS / ROS 与 Roban 下位机 HEFT 控制器通信；client 经 `kuavo_deploy/eval.py` 启动推理。
- **重要区分：** 通用 Kuavo 双臂数据准备、训练和部署文档仍是主流程；该文只描述 Roban whole-body 适配差异。

## 数据集适配注意

这份教程展示的是从 rosbag 开始的转换和训练步骤。它与 OpenLET `LET-HumanMotion-Roban2` 同属 Roban/全身动作生态，但教程没有明确说明数据包仓库中每个文件可直接喂入该脚本。取得申请权限后应先核验数据字段、版本和划分，再对齐 23D state / 32D action / 相机观测等模态配置。

## 对 wiki 的映射

- [LeTools 工具链](../../wiki/entities/letools.md)
- [OpenLET LET-HumanMotion-Roban2](../../wiki/entities/openlet-humanmotion-roban2.md)
- [OpenLET 社区](../../wiki/entities/openlet.md)
