## [2026-11-13] ingest | Kimodo-Policy 官方资源与源码闭环核验

- 意图：复用现有 `wiki/entities/paper-kimodo-policy.md`，避免重复项目节点；补全论文全体作者署名。
- 一手来源：arXiv:2611.11554v1 全文（既有归档已通读）；本次增量核对官方 GitHub README、训练/评测核心入口与真机部署脚本。
- 开源结论：官方发布了预训练、HumanoidArena、SIMPLE、真机 G1 + Dex3 数据/检查点/部署流程；BiGym 2.0、HumanoidToolBench 仍为计划项。
- 写回：`wiki/entities/paper-kimodo-policy.md`、`sources/papers/kimodo_policy_primary.md`、`sources/repos/kimodo-policy-official.md`；区分模型生成、SONIC 跟踪、两套仿真环境与实测成绩；记录未独立复现。
