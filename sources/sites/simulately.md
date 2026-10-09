# Simulately 官方网站

> 来源归档（ingest）

- **标题：** Simulately — Handy information and resources for physics simulators for robot learning research
- **类型：** site / project-page
- **URL：** <https://simulately.wiki/>
- **文档入口：** <https://simulately.wiki/docs/>
- **代码：** [RoboVerseOrg/Simulately](https://github.com/RoboVerseOrg/Simulately)（Apache-2.0，站点源码）
- **实验代码与数据：** 仓库 [`code` 分支](https://github.com/RoboVerseOrg/Simulately/tree/code)，About 页说明其中包含文档实验（如 rendering、getting-started）使用的代码与数据
- **相关来源：** [Simulately 仓库归档](../repos/simulately.md)
- **入库日期：** 2026-10-09
- **沉淀到 wiki：** 是 → [`wiki/entities/simulately.md`](../../wiki/entities/simulately.md)

## 项目页核查

主页将 Simulately 定位为机器人学习研究的仿真器信息与资源集合，主要内容包括仿真器介绍/比较、使用片段、相关工作、工具包及协作贡献入口。文档目录也覆盖演示数据集与物体、场景资料。当前可见文档包含 MuJoCo、Isaac Gym/Sim、SAPIEN、PyBullet、Gazebo、CoppeliaSim、Genesis、Taichi 等专题。

首页显示站点由 Docusaurus 构建，并由 Cloudflare 提供服务；README 给出的本地流程为 Node.js 18+、`npm install`、`npm run start`。这些入口运行的是网站本身，不会代替安装各个物理仿真器。

## 可用性与注意事项

- 项目源代码公开；Apache-2.0 是 Simulately 仓库的许可证信息。
- About 页提供了实验代码与数据的 `code` 分支链接，复用前应检查该分支下具体实验目录、依赖版本与脚本。
- 站内性能比较和 simulator 信息存在更新时间差异；使用 GPU/FPS 或受欢迎度数字做选型时，应查看原始测试代码、硬件条件与发布日期。
- Simulately 是 RoboVerseOrg 下的资源站项目，不等同于 RoboVerse 论文中描述的 MetaSim 通用仿真平台。

## 映射到知识库

该站适合作为仿真器知识检索和上手入口；它自身不定义统一的仿真执行 API，也不改变各引擎原有的许可证和接口。

