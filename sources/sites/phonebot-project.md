# PhoneBot 项目页（phonebot.dev）

> 来源归档

- **标题：** PhoneBot: A Low-Cost Open Humanoid Robot Platform Reusing Smartphones
- **类型：** site（项目页）
- **URL：** <https://phonebot.dev/>
- **论文：** <https://arxiv.org/abs/2610.08737>（归档：[phonebot_arxiv_2610_08737.md](../papers/phonebot_arxiv_2610_08737.md)）
- **硬件资料：** 项目页提供 BOM、Onshape CAD、STEP、可打印部件与装配教程下载入口。
- **硬件许可：** CAD、机械导出文件、BOM 与装配文档标注 CC BY-NC 4.0；商业使用需另行取得权利人许可。
- **代码：** 项目页说明手机端应用支持步态策略推理、视觉跟踪、SLAM 和交互；截至核查日，页面未提供可核验的软件源码仓库链接。论文的软硬件开源表述不能替代可访问的软件代码入口。
- **机构：** 加州大学洛杉矶分校（UCLA）RoMeLa
- **核查日期：** 2026-10-22
- **一句话说明：** 约 48.3 cm、13 DoF 的小型双足平台，以 Android 手机承担感知与高层推理，并通过 Wi-Fi 将关节目标发送至板载控制器。

## 当前开放状态

**硬件资料可获取，限非商业用途；软件源码尚未核实。** 相比 2026-10-12 的项目页快照（仅占位硬件图），当前页面新增了 CAD、BOM、STEP、打印件和装配教程资源入口，并明确硬件资料采用 CC BY-NC 4.0。公开页面未给出软件 GitHub/Hugging Face 仓库链接，因此不建立 sources/repos 代码仓条目，也不将软件栈描述为已验证可下载。

## 系统架构补充

- 手机提供相机、IMU、无线通信与应用层计算；项目页描述的 Android 应用包括视觉跟踪、SLAM、步态策略推理和用户交互。
- 手机通过 Wi-Fi 与机身单板机交换命令和遥测；单板机通过 U2D2 与 TTL 总线控制 13 个 DYNAMIXEL XL430-W250-T 执行器。
- 论文称语音识别与语言模型由笔记本后端承担，不能据此表述为完全离线的手机端对话。

## 对 wiki 的映射

- [PhoneBot 论文实体页](../../wiki/entities/paper-phonebot-smartphone-humanoid.md)

## 一手入口

- [PhoneBot 项目页](https://phonebot.dev/)
- [arXiv 论文](https://arxiv.org/abs/2610.08737)
- [CC BY-NC 4.0 许可](https://creativecommons.org/licenses/by-nc/4.0/)
