# MengTo/Skills — 设计与开发 Agent Skills 库

- **类型：** repo / 官方技能说明与脚本
- **作者：** Meng To
- **代码：** https://github.com/MengTo/Skills
- **核对版本：** [5ea1661a0e05d82b054553daaa6baf6d0233c33c](https://github.com/MengTo/Skills/tree/5ea1661a0e05d82b054553daaa6baf6d0233c33c)
- **核对日期：** 2026-11-08
- **核对范围：** 已通读官方 README；通读 agent-skills/codex/iterate-until-verified/SKILL.md、agent-skills/web-design/threejs/SKILL.md、agent-skills/workflow/README.md；抽查 scripts/validate-skill-demos.mjs 的入口、文件检查与校验循环。未逐个研读全部 176 个 Skill，也未实际执行截图校验。
- **通读：** 已完成
- **开放状态：** Skills Markdown、示例、脚本公开，MIT；不是一个统一训练或推理模型。
- **归入 wiki：** [MengTo/Skills](../../wiki/entities/mengto-skills.md)

## 官方结构与选取样本

README 在所核对提交中声明 **176 skills、9 类**；数字是版本快照，不宜当长期固定指标。最小单元 `agent-skills/<category>/<skill>/SKILL.md`，可选 `REFERENCES.md`、`ARTICLE.md`、`assets/`、`scripts/` 与 `demo/`。可被 Codex/Claude/Cursor 等 agent 按需读取。

- `iterate-until-verified`：锁定任务契约 → 可观察验收门 → 分工 → 独立核验 → 失败修复迭代 → 基于证据停止。
- `web-design/threejs`：Scene/Camera/Renderer、渲染循环、glTF、设备像素比、资源销毁与降级。
- `workflow/README.md`：截图、质量评分、发版和线程管理的组合式操作流程。
- `scripts/validate-skill-demos.mjs`：读取 git 跟踪的 SKILL.md 列表，遍历每个 skill 的 demo 资源，验证演示文件、预览图尺寸、脚本/资源等并输出错误；这属于 demo 静态检验脚本，不能当作所有 skill 功能测试通过的证明。

## 对机器人栈的意义

可借鉴「需求约束 → skills 工作流 → 检查证据 → 版本化复用」组织 CAD/URDF/场景设计或机器人实验的开发过程。但其示例以 Web/游戏/视觉设计为主，不提供现成的机器人闭环控制安全保证。

## 一手链接

- [根 README](https://github.com/MengTo/Skills/blob/5ea1661a0e05d82b054553daaa6baf6d0233c33c/README.md)
- [iterate-until-verified](https://github.com/MengTo/Skills/blob/5ea1661a0e05d82b054553daaa6baf6d0233c33c/agent-skills/codex/iterate-until-verified/SKILL.md)
- [Three.js Skill](https://github.com/MengTo/Skills/blob/5ea1661a0e05d82b054553daaa6baf6d0233c33c/agent-skills/web-design/threejs/SKILL.md)
- [demo 校验脚本](https://github.com/MengTo/Skills/blob/5ea1661a0e05d82b054553daaa6baf6d0233c33c/scripts/validate-skill-demos.mjs)
