## [2026-11-22] ingest | sources/repos/humancalib.md — 走路人体自动标定多相机外参并接入动捕路线

- 新增独立 HumanCalib 双语项目页，基于官方 METHOD、HOWTO 和 CLI/BA/后处理源码核实流程、运行时序与许可。
- 关键边界：免棋盘格外参标定，仍要求同步视频、已知内参、步行主体和真实身高；输出 Pose2Sim TOML 而非人体动作。
- 更新 FreeMoCap 交叉引用和 Motion Retargeting Stage 2 数据采集路径。
