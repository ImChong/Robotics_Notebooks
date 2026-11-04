# TACOS 官方仓库

> 来源归档（repo）

- **标题：** TACOS: Terrain-Aware COntact Supervision for Visual Footstep Planning
- **代码：** <https://github.com/norlab-ulaval/TACOS>
- **论文：** <https://arxiv.org/abs/2610.37806>
- **入库日期：** 2026-10-30；核对日期：2026-11-04
- **许可：** 代码 MIT；运行时下载的 DINOv3 权重适用 Meta 的独立许可。
- **沉淀到 wiki：** [TACOS](../../wiki/entities/paper-tacos.md)

## 官方入口及范围

README 和 pyproject.toml 提供 `tacos-download`（GrandTour 数据）、`tacos-labels`（支撑期评分与投影）、`tacos-train`、`tacos-evaluate`、`tacos-predict`。已核查标签、训练、模型和评估实现，详见实体页时序图。README 所列内容主要是标签、模型训练与预测；真机规划和控制部署代码不在已核查范围内。

快速示例依次下载 KAB-1、KAB-2，生成标签，指定 KAB-2 作为测试运行并训练、评估。自有数据需适配 README 所列的关节、IMU、接触、位姿、相机与图像格式；机器人运动学位于 `tacos/robot.py`。需要 Python 3.12，环境和权重细节以官方 README 为准。
