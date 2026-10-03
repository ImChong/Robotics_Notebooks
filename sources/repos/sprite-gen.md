# aldegad / sprite-gen

- **标题：** sprite-gen
- **类型：** repo
- **来源：** aldegad
- **链接：** <https://github.com/aldegad/sprite-gen>
- **入库日期：** 2026-10-03
- **一句话说明：** Apache-2.0 许可的 Python CLI 与 Codex / Claude skill，将单张角色图或视频转换为带透明通道、帧布局清单的 2D 游戏精灵图集与循环动画素材。
- **沉淀到 wiki：** 暂不。项目聚焦游戏美术资产，不涉及机器人模型、仿真、控制或训练数据；目前作为相邻的图像生成与资产处理工具留档。

## 开源状态核查（2026-10-03）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — GitHub 公开仓库，含 Python 包、CLI、脚本、skill 文档、Web curation 工具与测试 |
| Stars / Forks（API） | 约 2,328 / 246 |
| 默认分支 | `main` |
| 主要语言 | Python |
| 版本 | `pyproject.toml` 标注 2.19.0 |
| 许可 | Apache-2.0 |
| Python 版本 | 3.11+ |
| 项目页 | GitHub README（仓库未设置独立官网） |

## README 摘要

项目把从参考图到可供游戏引擎使用的精灵素材处理拆成可复用 CLI 阶段：按动作状态生成帧、清理色键背景并提取透明帧、合成图集与机器可读的 `manifest.json.frame_layout`。另有视频生成到透明循环、逐帧挑选与修正、调色、分层合成、切片、图集解包和场景合成等工具。

## 流程与接口

- **图集行流程：** 准备角色与动作列表 → 按状态生成帧 → 提取并清理 alpha → 合成 atlas 与 frame layout 清单 → 可选在 curation 页面检查、筛选和修正。
- **视频循环流程：** 准备视频画布 → 生成动作视频 → 抽帧 → 检查并组成无缝循环。
- **引擎衔接：** 产物是 PNG / GIF / 视频及帧布局元数据；使用方可依 manifest 将素材导入自有运行时。它不代替引擎本身的动画状态机，也不自动保证生成帧的角色一致性或动作正确性。

## 对机器人研究与工程的边界

仓库面向 2D 游戏素材，不是机器人资产建模、URDF/MJCF/USD 制作或仿真场景生成工具。生成出的二维动画可用于解释、展示或界面原型；不应将其当作机器人运动学真值、控制参考轨迹或具身训练数据。若研究目标是仿真可用的 3D 场景资产，应优先评估具备 3D 网格与格式导出的工具。

## 参考链接

- 仓库与英文 README：<https://github.com/aldegad/sprite-gen>
- 中文 README：<https://github.com/aldegad/sprite-gen/blob/main/README.zh-Hans.md>
- 流程架构说明：<https://github.com/aldegad/sprite-gen/blob/main/docs/architecture.md>
- 图集工作流：<https://github.com/aldegad/sprite-gen/blob/main/docs/atlas-workflow.md>
- 用户工作流：<https://github.com/aldegad/sprite-gen/blob/main/docs/user-workflow.md>
- 演示视频：<https://youtu.be/zVu9YlbPtog>
- 许可文本：<https://github.com/aldegad/sprite-gen/blob/main/LICENSE>
