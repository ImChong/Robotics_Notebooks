# Spectacles Dimensional OS | OpenYAM AR 遥操作源码归档

- **作者 / 组织：** Johannes Tscharn（GitHub `V4C38`）及贡献者；此为个人仓库项目，不把作者雇主视为项目发布方。
- **代码：** [V4C38/dimensional-os-ar](https://github.com/V4C38/dimensional-os-ar) · [固定 development 提交 9e0e103](https://github.com/V4C38/dimensional-os-ar/tree/9e0e103773cea967536534b528512e3c50e7be86)
- **版本边界：** 旧仓库地址 `V4C38/spectacles-dimensional-os` 重定向到现名；两地址是同一项目。2026-11-26 在官方 Releases 与 git tags 中查到 `v1.0.0` tag；未查到 `v2.0.0`。`main` 的 README 仍以 Go2/G1 为主，OpenYAM 使用公开的 `development` 路径。仓内依赖固定 DimOS `28dc6f2714484f9011218f3667007d740215d358`。
- **许可：** 根目录 MIT；OpenYAM 网格及派生视觉资产另见 [模型来源说明](https://github.com/V4C38/dimensional-os-ar/blob/9e0e103773cea967536534b528512e3c50e7be86/clients/webxr/public/models/NOTICE.md) 和上游许可。
- **项目及演示：** [作者原帖与来源核对](../sites/spectacles-openyam-demo.md)；[DimOS OpenYAM 上游集成文档](https://docs.dimensional.org/capabilities/manipulation/openyam_integration/)。
- **核对范围：** 2026-11-26 阅读 development README、AR manipulation 指南、协议的定位/共用意图/动作所有权段、Specs 手势输入、客户端共享模块、OpenYAM 蓝图及两类运动通路、bench 预设与测试入口。在 Node 24（非项目指定的 Node 22）下运行 ClientCore 单元测试，57 文件、638 测试通过；未运行原生 Lens、装载 DimOS mock 合同或访问真机。
- **通读：** 已完成
- **对应节点：** [Spectacles × DimOS：OpenYAM AR 遥操作](../../wiki/entities/spectacles-dimensional-os.md)

1. [SpecsManipulationController.ts](https://github.com/V4C38/dimensional-os-ar/blob/9e0e103773cea967536534b528512e3c50e7be86/clients/specs/Assets/Scripts/DimOSSpecsClient/manipulation/SpecsManipulationController.ts) 走 Spectacles Interaction Kit 的双手捕获、坐标转换和抓取偏置。
2. [PROTOCOL.md](https://github.com/V4C38/dimensional-os-ar/blob/9e0e103773cea967536534b528512e3c50e7be86/dimos-ar/PROTOCOL.md#shared-following-transport) 定义共享目标、编辑租约、服务器所有权和姿态反馈；同一目标不由客户端再发第二条 native stream。
3. [blueprints.py](https://github.com/V4C38/dimensional-os-ar/blob/9e0e103773cea967536534b528512e3c50e7be86/dimos-ar/dimos/ar/manipulation/blueprints.py) 连接 ARModule、ARManipulationModule、DimOS 协调器和 OpenYAM 硬件；设置 `ar_cartesian_ik` 跟随任务。
