# Google DeepMind robotics-antigravity-plugin 官方源码

- **类型：** repo / agent robotics plugin
- **机构：** 谷歌 DeepMind（Google DeepMind）
- **代码：** https://github.com/google-deepmind/robotics-antigravity-plugin
- **源码版本：** 0.1.13，`0f640d8622c5d6335f068927d68a25958d0dda9a`
- **插件文档：** https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/README.md
- **相关模型主页：** https://deepmind.google/models/gemini-robotics/
- **核查日期：** 2026-11-28
- **核对范围：** 通读固定提交的根 README、插件 README、发现/版本清单及相关安全、工位标定、抓放、RGB 定位和视觉伺服技能，并核对标定与夹爪辅助实现；未安装插件、运行外部感知服务、连接真机或审计全仓库
- **通读：** 已完成

## 原文记录

仓库提供插件发现清单、技能协议、Python 辅助函数、演示与安全规则。0.1.13 新增通用 `workspace-calibration` 和 `pick-and-place`，删除旧的独立 `so101-calibration` 技能；README 的饼干分拣展示是 YAM Pro 的 MuJoCo 和真机两条演示路径，不提供通用任务成功率。安装来源决定 MuJoCo API 技能是否已经捆绑；第三方软件、机器人模型及 Gemini 服务授权独立于插件的 Apache-2.0。

## 核对入口

- [插件元数据](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/plugin.json)与[发现清单](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins.json)：本次版本及内容边界。
- [工位标定技能](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/skills/workspace-calibration/SKILL.md)与 [calibration.py](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/skills/workspace-calibration/scripts/calibration.py)：缓存分开记录感知平面和控制命令；仿真和实物不共用缓存，图像边缘指纹只能做快速失效检查。
- [抓放技能](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/skills/pick-and-place/SKILL.md)与 [grasp_helpers.py](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/skills/pick-and-place/scripts/grasp_helpers.py)：辅助函数通过外部回调读写夹爪；按夹爪开闭方向判断相对于空夹基线的余量，不能仅据此判定最终摆放。
- [物理机器人安全规则](https://github.com/google-deepmind/robotics-antigravity-plugin/blob/0f640d8622c5d6335f068927d68a25958d0dda9a/plugins/robotics/rules/physical_robot_safety.md)：退场、速限、心跳和温度约束是需要按硬件校验的规则，不是独立硬件安全控制器。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/google-deepmind-robotics-antigravity-plugin.md)
