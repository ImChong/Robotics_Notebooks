# Jujutsu（jj-vcs/jj）

> 来源归档（ingest）

- **类型：** repo / version control system
- **主仓库：** <https://github.com/jj-vcs/jj>
- **许可证：** Apache-2.0（仓库 LICENSE）
- **入库日期：** 2026-10-21
- **开源状态：** 已开源；公开 Rust 源码、文档与安装说明。
- **一句话说明：** Jujutsu（命令 `jj`）是兼容 Git 存储的版本控制工具，以可撤销操作、自动追踪工作区改动和无暂存区工作流简化提交管理。
- **沉淀到 wiki：** 是 → [Jujutsu 实体页](../../wiki/entities/jujutsu.md)

## 核心入口

- README 与快速开始：<https://github.com/jj-vcs/jj>
- 官方文档：<https://docs.jj-vcs.dev/latest/>
- 安装与 Git 协作：<https://docs.jj-vcs.dev/latest/install-and-setup/>
- 许可证：<https://github.com/jj-vcs/jj/blob/main/LICENSE>

## 仓库要点

| 面向 | 归纳 |
|------|------|
| 实现 | Rust 编写的版本控制工具，命令行入口为 `jj` |
| 存储 | 默认可使用 Git 仓库作为后端，与既有 Git 仓库及工具互通；jj 自己负责工作区、提交和操作历史等用户界面与逻辑 |
| 工作流 | 不要求显式暂存区；工作区改动自动进入当前 change，提交可持续演化 |
| 可恢复性 | 操作记录可查询，误操作可通过 undo 等机制撤回；改变操作历史不代表远端协作操作都能无条件回滚 |
| 适用范围 | 个人仓库到团队协作均可使用；具体迁移方式应先在副本中验证团队的 Git hook、GUI 与 CI 工具链兼容性 |

## 使用边界

Git 后端降低了与 Git 生态互通的门槛，但 jj 是另一套命令与协作模型。团队采用前应检查自动化脚本、IDE、GUI 客户端、钩子和托管平台是否覆盖所需操作；涉及共享远端历史的改写仍需按团队约定处理。GitHub 的 star、fork 数会变化，本归档不固定记录计数。

## Wiki 映射

- [Jujutsu 实体页](../../wiki/entities/jujutsu.md)
- [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 参考来源

- [Jujutsu 官方 GitHub 仓库](https://github.com/jj-vcs/jj)
- [Jujutsu 官方文档](https://docs.jj-vcs.dev/latest/)
