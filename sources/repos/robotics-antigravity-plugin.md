# Google DeepMind robotics-antigravity-plugin 官方源码

- **类型：** repo / agent robotics plugin
- **代码：** https://github.com/google-deepmind/robotics-antigravity-plugin
- **插件文档：** https://github.com/google-deepmind/robotics-antigravity-plugin/blob/main/plugins/robotics/README.md
- **相关模型主页：** https://deepmind.google/models/gemini-robotics/
- **核查日期：** 2026-11-20
- **核对范围：** 根 README 及 plugins/robotics/README.md 完整文档、安装示例与安全提示；未逐个审计 skills/rules/sidecars
- **通读：** 已完成

## 原文记录

官方命令 `agy plugin install https://github.com/google-deepmind/robotics-antigravity-plugin`；可用 `npx skills add google-deepmind/robotics-antigravity-plugin` 导入单项技能，MuJoCo 独立技能以 `google-deepmind/mujoco` 为官方来源。目录有 `plugins/robotics/`、`rules/`、`sidecars/`、`skills/`，此插件是**智能体工作流插件，不是物理引擎或训练模型**。README 用 SO-101/YAM 机械臂及输送带分拣演示同一控制代码先仿真再真机；部分感知技能调用 Gemini API 需用户环境变量 `GEMINI_API_KEY`。警示：真机运行须有人在场、可断电、清空工作区域。Apache-2.0，不覆盖第三方软件/模型许可。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/google-deepmind-robotics-antigravity-plugin.md)
