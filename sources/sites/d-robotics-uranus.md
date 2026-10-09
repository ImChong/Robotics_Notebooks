# Uranus 官方项目页与技术博客

- **URL：** <https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/>
- **作者/团队：** 地瓜机器人（D-Robotics）大模型团队
- **博客页面日期：** 2026-08-25（网页显示）
- **关联论文：** [Uranus arXiv:2609.24815](../papers/uranus_arxiv_2609_24815.md)
- **代码：** [Uranus-OSS](../repos/uranus-oss.md)
- **核查日期：** 2026-10-09

## 项目定位

项目页将 Uranus 定位为面向具身 AI 的交互式生成式模拟基础设施。它以机器人动作轨迹、机器人结构与相机几何为条件生成未来多视角视频，用于闭环策略评估与数据扩充。论文 v3 于 2026-09-23 更新；页面所示博客日期为 2026-08-25，二者记录的是不同页面/版本日期。

## 项目页信息摘录

- 作者报告训练数据规模为约 3,300 小时机器人操作视频，并给出 64 张 GPU 的训练实现。
- 通过 Ray/Daft/Lance 数据管线、Rust I/O 与分布式训练优化，报告端到端生成速率 24 FPS。
- 项目页链接至论文与开源入口；实际可复现内容以公开 Uranus-OSS README、demo 数据与模型卡为准，不能据此宣称完整训练配方或全量数据全部开源。

**对 wiki 的映射：** [Uranus 项目节点](../../wiki/entities/paper-uranus.md)。
