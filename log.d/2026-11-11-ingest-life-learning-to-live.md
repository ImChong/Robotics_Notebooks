# 2026-11-11 — ingest LIFE：Learning to Live

- 来源：arXiv:2611.00180v2（2026-11-04）、AgibotTech/LIFE 官方 README、RoboBeats `gate.py`、SSS-Bench `pipeline.py` / `agent.py`，项目页以及前作 ReSight arXiv:2610.33487。
- 唯一主实体：[`wiki/entities/paper-life-learning-to-live.md`](../wiki/entities/paper-life-learning-to-live.md)。
- 区分论文完整系统主张（RoboBeats / Plastic Memory / EDM）与公开可运行部分（门控推理、模型适配器、SSS-Bench）。
- 按源码绘制因果事件流图和 `Gate.stream` → `select` → writer / reader 的 Mermaid 运行时序图；解释 100/10/90 分块、0.5/0.3 滞回、提问截断与 hard10 单次最多 10 帧回看。
- 从既有 Agent 记忆 wiki 建反向链接；未改动 catalog/log/export 派生文件。
- 论文数字属作者报告，未进行 CUDA benchmark 与实机复现；项目主页正文抓取未完成，具体边界记录在 sources/sites。
