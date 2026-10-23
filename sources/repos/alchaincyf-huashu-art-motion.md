# huashu-art-motion（编码代理艺术动画 Skill）

> 来源归档（repo；核对日期：2026-10-23）

- **仓库：** <https://github.com/alchaincyf/huashu-art-motion>
- **类型：** 面向编码代理的动画制作 Skill 与可复制代码工程。
- **用途：** 将程序化艺术场景、解说动画语法、参数化片段和视频验收工具组织成一套制作流程；README 还提供拆解参考动画的工具与长卷示例。
- **使用入口：** npx skills add alchaincyf/huashu-art-motion。完整 Skill 在仓库根目录 SKILL.md，使用者按任务读取 references/ 中的方法说明，再在项目副本中运行 scripts/engine/。
- **主要组件：** references/ 收录风格配方、动画语法与制作方法；scripts/engine/ 提供场景、渲染器、示例和参数化片段；scripts/analyze/breakdown.py 提取参考视频的切点、节拍与运动线索；scripts/qa.py、scripts/film_gate.py、scripts/deliver.py 用于检查和交付渲染结果。
- **运行依赖：** README 列出 uv、FFmpeg 和 Playwright Chromium；具体脚本依赖随工作流而异。渲染视频需要配置对应依赖。
- **开放与许可：** 仓库公开，README 声明代码和文档采用 MIT。许可例外包括衍生笔顺数据、各自保留许可的字体，以及作者角色形象/帧素材和 Clawd 同人示例；使用前应按仓库许可证说明分别核查素材。
- **适用边界：** 它是动画制作与技术解释工具，不是机器人仿真器，也不提供动力学、碰撞或控制正确性的验证。生成画面应与仿真结果、原始数据和论文内容分开核验。
- **资料依据：** 本条根据上游 README、SKILL.md、LICENSE 与仓库目录说明整理；项目能力和许可细节以当前上游文件为准。

## 互链

- [Wiki 实体](../../wiki/entities/huashu-art-motion.md)
- [上游仓库 README](https://github.com/alchaincyf/huashu-art-motion#readme)
- [上游 SKILL.md](https://github.com/alchaincyf/huashu-art-motion/blob/main/SKILL.md)
