# FreeMoCap

> 来源归档

- **标题：** FreeMoCap
- **类型：** repo / 软件平台
- **来源：** freemocap 组织（GitHub）
- **链接：** https://github.com/freemocap/freemocap
- **PyPI：** https://pypi.org/project/freemocap/
- **官方文档：** https://freemocap.github.io/documentation
- **入库日期：** 2026-05-12
- **许可证：** AGPL-3.0（README 说明若 AGPL 不适用可联系项目方商议其他授权条款）
- **一句话说明：** 面向科研与教学的 **开源、低成本、软硬件相对agnostic** 的多相机运动捕捉系统与 GUI 平台，主打「最小成本获得研究级动捕管线」。
- **沉淀到 wiki：** 是 → [`wiki/entities/freemocap.md`](../../wiki/entities/freemocap.md)

---

## 核心定位（来自 README 的公开主张）

- **硬件与软件 agnostic**：强调在常见 USB 相机等低成本硬件上工作，降低动捕门槛。
- **研究级**：面向 decentralized scientific research, education, and training 的定位自述。
- **入口形态**：`pip install freemocap` 后通过 `freemocap` 或 `python -m freemocap` 启动 GUI；官方推荐 Python 3.10–3.12（README 写 3.12 recommended），源码安装示例使用 conda + `pip install -e .`。

---

## 文档与社区

- 安装细节见官方文档 Installation 章节（README 链接）。
- 新手录制流程见 *Beginner Tutorials*（官方文档）。
- 文档站点源码仓库：<https://github.com/freemocap/documentation>（Writerside）。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [retarget.md](../retarget.md) | 动捕数据进入机器人学习前通常需要重定向与清洗 |
| [motion.md](../motion.md) | 动捕数据集与运动生成资源索引 |
| [mjlab_playground.md](mjlab_playground.md) | 示例：动捕/参考轨迹经处理后，可作为模仿学习或奖励设计的输入，再在 mjlab 系任务中训练策略（非仓库内置一键链路，属于研究管线组合） |
