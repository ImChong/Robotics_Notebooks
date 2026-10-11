# Anatomy 官方仓库

- **标题：** Anatomy — Claude Code Agent Skill
- **类型：** repo / agent-skill / technical-illustration / svg / webgl / interactive-3d
- **维护者：** Ryan（GitHub: wheresryan22）
- **代码：** <https://github.com/wheresryan22/anatomy>（公开，MIT）
- **项目页：** <https://skills.wheresryan.sh/anatomy>
- **入库日期：** 2026-10-11
- **一句话说明：** Claude Code 技能：把技术原理转成有真实部件、交互读数和动画的等轴测机械插图，输出 SVG / HTML / React；可选 WebGL 特效，3D 动作模式为 beta。
- **知识节点：** [Anatomy](../../wiki/entities/anatomy-skill.md)

## 开源状态核查（2026-10-11）

| 项 | 核查 |
|----|------|
| **源码** | 公开 GitHub 仓；包含 `SKILL.md`、绘图 kit、参考指南、示例与验证脚本 |
| **许可** | MIT |
| **安装** | `npx skills add wheresryan22/anatomy -g -a claude-code`；也可将仓库放入 Claude Code skills 目录 |
| **运行依赖** | Claude Code 负责按 Skill 工作流生成；构建脚本使用 Node.js，核心 kit 无需额外 npm 依赖（React 输出例外） |
| **结论** | Skill 指令与生成/审计工具均公开；3D 模式仍标注 beta |

## 仓库结构与机制

| 路径 | 作用 |
|------|------|
| `SKILL.md` | 规定从真实参数、物理类比、零件清单到几何构建、交互与验证的生成流程 |
| `kit/` | 等轴测相机、实体与管线/圆件绘制、SVG 渲染、WebGL 层、动画运行时及 React 组件 |
| `references/` | 画面工艺、Kit API、动效、React、WebGL、3D 与验证指南 |
| `examples/` | 可直接构建的机械示例，附 HTML 与脚本 |
| `kit/audit.mjs` | 审核实体穿插、连接端点、装配状态和绘制深度顺序 |
| `scripts/capture.mjs`、`drive.mjs` 等 | 截图、驱动画面、检查悬空线段和 3D 图形 |

## 对 wiki 的映射

- 主实体：[Anatomy](../../wiki/entities/anatomy-skill.md)
- 关联概念：[文字生成 CAD（Text-to-CAD）](../../wiki/concepts/text-to-cad.md)
- 对照项目：[3D Gen Studio](../../wiki/entities/3dgenstudio.md)

## 参考链接

- [GitHub README](https://github.com/wheresryan22/anatomy/blob/main/README.md)
- [SKILL.md](https://github.com/wheresryan22/anatomy/blob/main/SKILL.md)
- [MIT License](https://github.com/wheresryan22/anatomy/blob/main/LICENSE)
