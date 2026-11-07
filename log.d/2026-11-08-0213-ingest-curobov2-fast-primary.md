## [2026-11-08] ingest | 深读 NVIDIA cuRoboV2-Fast 论文与源码并修正性能对比口径

- 复用既有 arXiv:2611.06913 节点，不重复建项目；论文、官网、官方代码三层一手归档复核。
- 核验 RTX 5090/3kg 全链路规划 4.43 ms、14.7×，Jetson 25.99 ms，YAM 真机周期 5.93 s/缩短 34.6%；明确规划时延不等于真实周期。
- 补 B-spline、GPU BiRRT、TimeOpt 原理、项目 Mermaid 流程图和基于官方 Python/TimeOpt 入口的 Mermaid 时序图。
- 官方 NVlabs/curobo 已开源（Apache-2.0）；仅静态读码，未本地复现性能。