## [2026-11-10] ingest | Guava（arXiv:2606.18363v3）：机器人语义 Harness 与 4B 蒸馏

- **意图：** 区分大模型逐步语义工具调用和底层实时控制，补足带失败恢复的数据蒸馏路线。
- **一手核对：** 通读论文 v3 正文/附录，核对项目页、发布仓库 README、评测主链、数据采集与恢复分支、SFT/GRPO 入口。
- **技术核心：** GPT-5.4 → 共享工具 Harness + MuJoCo 轨迹/恢复分支 → Qwen3.5-4B SFT；GRPO 仅附加于选定长任务。
- **数据口径：** 2,268 条成功及恢复轨迹（1,554 主轨迹/714 分支）；仿真 15×30 次 87.1%，真机 9×10 次 90.0%（论文作者报告）。
- **源码与权重：** 官方仿真执行、轨迹、训练及评测脚本公开；SAM3 权重与真实机器人条件另需具备。仓库根目录未发现顶层 LICENSE，授权不可臆断。
- **新增实体：** [Guava：机器人操作 Harness 中的前沿 VLM 蒸馏](../wiki/entities/paper-guava-harness.md)（含原文 Mermaid 流程图与源码运行时序图）。
- **站内关联：** [LLM 机器人控制接口](../wiki/concepts/llm-robotics-control-interfaces.md) 入链，区别于同名 3D 渲染 GUAVA。
- **测试边界：** 本次经 GitHub 静态源码核对；未启动 MuJoCo/模型推理或进行真机复现，完整 CI 以 PR Actions 为准。
