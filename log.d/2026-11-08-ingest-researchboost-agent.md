# 2026-11-08 ingest — ResearchBoost-Agent：可审计的 AI 训练 Infra 优化

- **意图：** 根据作者的《ResearchBoost: RSI starts with Infra》、官方 README/Methodology/Judge 及可运行脚本，收录“先冻结裁判再由 Agent 持续提效”的技术路线，并为机器人训练工程提供辨析。
- **材料：** `sources/blogs/researchboost-rsi-starts-with-infra-2026.md`；`sources/repos/researchboost-agent.md`。
- **主节点：** [ResearchBoost-Agent 双语深读](../wiki/entities/researchboost-agent.md)（一个项目一个实体，不为 Motus/NanoChat 重复建立 ResearchBoost 主节点）。
- **开源/源码结论：** GitHub 仓库公开了实际运行入口及评测组件；本次查到的根目录未列 `LICENSE`，未擅自认定宽松代码再分发许可。
- **核心核验：** 2026-11-08 官方 README 累计 20 项、7 个领域，9.59× 仅是 8×B200 上 Faster-Motus 7.468→0.779 秒/迭代；NanoChat 1.0744× 在 8×H20 的 FP8 训练更新中测得。必须区分 step time、estimated eval wall-clock 与完整训练。
- **流程可溯源：** 冻结数值 baseline σ → 逐轮 profile/time/check → FINAL.json → `judge.sh` 最终 B-C-B `check --final` → `accept.py` PASS/收益/哈希校验。节点含 Mermaid 项目流程图和真实源码入口时序图。
- **边界：** 尚未独立 GPU 复现；短轨迹验证、bubblewrap 回退安全和商业授权均已单独标注。研究 Infra 速度提升不等于完整 RSI。
- **图谱入链：** [递归自改进 RSI](../wiki/concepts/recursive-self-improvement.md) 添加反向入口。
- **变更边界：** 仅改 wiki、sources 与 log.d；未触碰 `catalog.md`、`log.md`、导出/站点统计文件。
