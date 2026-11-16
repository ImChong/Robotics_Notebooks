# Specialist — 官方项目与硬件文档核验

- **类型：** site / 官方文档
- **官网：** https://specialist-ai.github.io/
- **官方组织介绍：** https://github.com/specialist-ai/.github/blob/main/profile/README.md
- **构建文档：** https://specialist-ai.github.io/specialist-docs/
- **入库日期：** 2026-11-09
- **机构：** 加利福尼亚大学伯克利分校（UC Berkeley，官方组织 README）
- **代码：** https://github.com/specialist-ai/specialist
- **论文：** 官方组织资料标注 Paper (coming soon)，未找到可核实的论文全文
- **核对范围：** 已阅读官方 GitHub 组织说明及硬件构建文档 BOM、handheld、actuated、scene、dataset 各节，并逐项核对其仓库入口；项目官网为 JS 页面，正文不可完整提取，故不宣称已逐页通读官网全部展示内容。
- **通读：** 已完成

## 官方系统定位

Specialist 采用手持平行夹爪和消费级 Insta360 双鱼眼相机采集人类操作示范；夹爪上的 AprilTag、环境锚点与 IMU / 图像信息用于离线重建公制 6DoF 轨迹、手指夹角。机器人执行端使用手指形态和相机位置相匹配的主动夹爪。官方主张可减少运动重定向需求，但不能由此推断跨任意末端执行器零误差迁移。

## 硬件与环境部署

- **手持端：** 官方 BOM 列出 3D 打印结构、Insta360 X3、AprilTag、螺栓、弹簧等；参见 [BOM](https://specialist-ai.github.io/specialist-docs/bom/) 与 [手持装配](https://specialist-ai.github.io/specialist-docs/handheld/)。
- **主动端：** 打印件、STS3215 舵机与相机等，见 [主动夹爪装配](https://specialist-ai.github.io/specialist-docs/actuated/)。装配文档中舵机 ID/对中说明尚有未完成步骤，复现时不可视为已闭环。
- **场景：** [Scene](https://specialist-ai.github.io/specialist-docs/scene/) 要求固定的 AprilTag 锚点提供世界坐标及尺度。相机一侧拍摄环境锚点，另一侧记录操作视角；开始保持静止、结束敲击动作帮助界定采样窗口。

## 开源状态（2026-11-09）

| 部分 | 状态 | 证据 |
|---|---|---|
| CAD、URDF、MJCF 与装配说明 | 已公开 | https://github.com/specialist-ai/specialist_description |
| 视频 / IMU / 标签联合优化数据处理 | 已公开 | https://github.com/specialist-ai/specialist |
| 夹爪、Feetech 和 Insta360 驱动 | 已公开 | specialist_sdk、feetech_driver、insta360_driver |
| LeRobot 插件、BC 训练 / ONNX / TensorRT | **未公开（coming soon）** | 官方组织主页列 specialist_lerobot、specialist_bc |
| specialist_sim、specialist_arm | **仓库链接目前无法访问（404）** | 官方组织主页列出的两个 URL |
| 论文全文、正式实验结果 | **待发布** | 官方组织主页 Paper (coming soon) |

官网宣称“fully open-source”不等于所有模块当日都可获取；所有开放判断以可访问的官方代码仓库为准。数据录制代码不等于开源训练数据集，未见可验证公开数据集目录。

## 文档 / 源码版本差异

官方 [Dataset 文档](https://specialist-ai.github.io/specialist-docs/dataset/) 部分示例使用 `--session-id` 和 `state_s12_r100`；目前主仓 `scripts/convert.py` 不含 `--session-id` 参数，`scripts/state.py` 默认 `--scale 8`，对应 `state_s8_r100`（360×360）。实际运行优先检查所用 commit 的 `--help`，不能无差别复制旧文档命令。

## 对 wiki 的映射

- [Specialist 项目主节点](../../wiki/entities/specialist-umi.md)
- [源码及调用链](../repos/specialist-ai-specialist.md)
