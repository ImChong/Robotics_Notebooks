## [2026-11-14] ingest | sources/sites/steel-skild-ai.md — 复核 Steel 的完整异构基准与复现实验边界

- 2026-11-09 Skild AI × NVIDIA 官方发布的 Steel 已于先前 ingest 建立唯一实体；本次复用原页，不重复建节点。
- 核对并补录 5/10/15/20 种混合任务吞吐表、20 类任务 matched-batch 12.0×、约 62% 显存节约。
- 澄清 `reuse_kinematics=True` 仅用于单任务对照，异构批次因 joint velocity limits 未启用；补充 PhysX 6.1 与 6.0 的版本敏感性、5 ms 物理步和单卡评测条件。
- 更新 Newton 求解器目录和 Steel 双向知识链接；性能数字均源于作者发布，未独立运行基准。
