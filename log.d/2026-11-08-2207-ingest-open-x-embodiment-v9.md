## [2026-11-08] ingest | sources/papers/open-x-embodiment-arxiv-2310-08864-v9.md — 通读 OXE v9，升级既有 RT-X 节点

- 官方项目主页是 `robotics-transformer-x.github.io`，源码为 `google-deepmind/open_x_embodiment`；复用既有论文实体，避免重复。
- 区分 60 个数据集/22 种机器人本体/1M+ 轨迹与论文实验实际使用的 9 种机械臂混合数据。
- 基于论文 IV–V 和官方 RT-1-X 离线推理示例复核 Mermaid 主流程/源码时序；修正 61% 与 62% 的泛化评测口径。
- 开放范围：数据读取示例和 RT-1-X checkpoint / 推理示例；不把 RT-2-X 55B 全链路训练或真机部署写成开源。
