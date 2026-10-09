# RoboVerseOrg/Simulately

> 来源归档（ingest）

- **标题：** Simulately — A universal summary of current robotics simulators
- **类型：** repo / documentation website
- **仓库：** <https://github.com/RoboVerseOrg/Simulately>
- **官方站点：** <https://simulately.wiki/>
- **默认分支：** `main`
- **许可证：** Apache-2.0（GitHub 仓库元数据）
- **项目页：** [Simulately 官方网站归档](../sites/simulately.md)
- **入库日期：** 2026-10-09
- **沉淀到 wiki：** 是 → [`wiki/entities/simulately.md`](../../wiki/entities/simulately.md)

## 仓库内容与用途

README 将项目描述为机器人学习研究使用的机器人/物理仿真器信息网站。主分支包含 Docusaurus 文档站：仿真器说明与对比、blog、操作 snippets、工具包、数据集/场景资料和站点配置。它是资料站源码，不是物理仿真引擎。

站点 [About 页面](https://simulately.wiki/docs/) 还指明，文档中实验（例如 rendering 与 getting-started）所用的代码和数据放在同一仓库的 [`code` 分支](https://github.com/RoboVerseOrg/Simulately/tree/code)。该分支与主分支的网站源码用途不同；使用实验材料时应检查相应脚本与测试条件。

## 本地运行

README 列出的流程：

```bash
npm install
npm run start
```

README 推荐 Node.js 18 或以上。仓库的 `package.json` 使用 Docusaurus 2.4.3，并定义 `start` 和 `build` 脚本；本地启动的是文档网站。

## 复现与时效提示

- 仓库内容横跨多个年份，具体 simulator 页面、API 示例与性能比较须按引擎版本复核。
- 比较页中的硬件结果受 GPU、驱动、配置及脚本影响；站点也提示这些比较并非普适权威结论。
- 站点代码采用 Apache-2.0，不代表所链接的仿真器、数据集或模型都具有相同许可。
- 本归档记录仓库与站点的关系，不把 RoboVerse 论文或 RoboVerse 仿真平台并入 Simulately 的项目身份。

## 关联资料

- 官方站点内容：[`sources/sites/simulately.md`](../sites/simulately.md)
- 项目实体节点：[`wiki/entities/simulately.md`](../../wiki/entities/simulately.md)
