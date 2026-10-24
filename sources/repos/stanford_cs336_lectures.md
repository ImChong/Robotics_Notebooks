# Stanford CS336 Lectures（讲义仓库）

> 来源归档（ingest）

- **仓库：** <https://github.com/stanford-cs336/lectures>
- **类型：** repo / course materials
- **对应课程：** [Stanford CS336 · Language Modeling from Scratch](../courses/stanford_cs336_language_modeling.md)
- **课程站：** <https://cs336.stanford.edu/>
- **课程站归档：** [sources/sites/stanford_cs336_language_modeling.md](../sites/stanford_cs336_language_modeling.md)
- **入库日期：** 2026-10-24
- **一句话说明：** Stanford CS336 Spring 2026 讲义仓库，包含 lecture_XX.py 可执行讲座和 lecture_XX.pdf 非可执行讲义。

## README 运行入口

README 描述的可执行讲座流程：

1. 使用 uv sync 安装 Python 环境，并克隆依赖 percyliang/edtrace。
2. 运行 python execute.py -m lecture_01 生成 var/traces/lecture_01.json。
3. 启动 edtrace/frontend 开发服务器，在浏览器中查看 trace；构建前端资源可用于部署到课程站。

仓库公开可读；GitHub 仓库元数据未返回许可证信息，因此不据此推断讲义或代码的授权条款。

## 参考入口

- [课程讲义仓库](https://github.com/stanford-cs336/lectures)
- [CS336 官方课程站](https://cs336.stanford.edu/)
