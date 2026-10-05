# Spingi Viewer（项目页）

> 来源归档（项目页 / 开放状态核查）

- **标题：** Spingi Viewer
- **类型：** static web viewer / episode replay
- **项目页：** <https://spingi-viewer.netlify.app/>
- **代码：** <https://github.com/ceccode/spingi>（`viewer/` 与 `runtime/`）
- **配套运行时：** <https://github.com/ceccode/spingi/tree/main/runtime>
- **Episode 格式：** <https://github.com/ceccode/spingi/blob/main/docs/episode-format.md>
- **入库日期：** 2026-10-05
- **一句话说明：** 浏览器端 3D episode 回放器，显示 G1 与对象轨迹、事件时间线、相机帧、安全和人工操作事件。

## 项目页核查

项目页已打开，页面标题为 “Spingi Viewer”，提供 ZIP 加载与内置样例入口。官方 Viewer README 说明页面是静态站点：episode 在浏览器内读取，不会上传；样例 episode 随仓库发布。

| 项 | 核查结果 |
|----|----------|
| 源码 | **已开源** — Runtime 与 Viewer 均在 GitHub 仓库 |
| Viewer 依赖 | TypeScript / Three.js；无需运行仿真器即可回放已记录 episode |
| 数据/权重 | 未发现模型权重或独立训练集；仓库含演示 episode 样例 |
| 真机控制 | 仓库当前提供 FakeAdapter 与 MuJoCo G1 SimAdapter；真机 adapter 是后续计划，不应按真机系统使用 |
| 许可 | Runtime 仓库 Apache-2.0；G1 模型资产另附 BSD-3-Clause notices |
| 项目性质 | Viewer 是 Runtime 的 episode 阅读端，不是单独的在线机器人控制面板 |

## 对 wiki 的映射

- [Spingi 实体页](../../wiki/entities/spingi.md)
- [仓库来源归档](../repos/spingi.md)
