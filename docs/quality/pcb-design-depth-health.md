# PCB 设计纵深路线节点健康度复核

- 复核日期：2026-11-16。
- 范围：路线页全部直接 wiki 链接，含页尾导航；41 个唯一节点，现全部为 **3/3**（首次检查 6/41 满分）。
- 实际 Stage 挂载：36 个唯一节点，全部 **3/3**；路线页自身 **3/3**。
- 沿用 `generate_link_graph.compute_health`；未修改算法、删除低分节点或虚填无法读取的原文。

## 实测结果

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/concepts/can-fd.md | 3/3 | — |
| wiki/concepts/ethercat-protocol.md | 3/3 | — |
| wiki/concepts/field-oriented-control.md | 3/3 | — |
| wiki/concepts/humanoid-mass-production-engineering.md | 3/3 | — |
| wiki/concepts/humanoid-testing-workflow.md | 3/3 | — |
| wiki/concepts/humanoid-wire-harness-types-protection.md | 3/3 | — |
| wiki/concepts/joint-encoder-selection.md | 3/3 | — |
| wiki/concepts/joint-module-electromagnetic-isolation.md | 3/3 | — |
| wiki/concepts/micromouse.md | 3/3 | — |
| wiki/concepts/robot-onboard-communication-architecture.md | 3/3 | — |
| wiki/concepts/robot-power-distribution-architecture.md | 3/3 | — |
| wiki/concepts/rs-232-serial-interface.md | 3/3 | — |
| wiki/concepts/rs-485-serial-bus.md | 3/3 | — |
| wiki/concepts/ttl-serial-logic-level.md | 3/3 | — |
| wiki/concepts/uart-serial-communication.md | 3/3 | — |
| wiki/entities/altium-designer.md | 3/3 | — |
| wiki/entities/atopile.md | 3/3 | — |
| wiki/entities/heypcb.md | 3/3 | — |
| wiki/entities/kicad.md | 3/3 | — |
| wiki/entities/konnect.md | 3/3 | — |
| wiki/entities/moteus.md | 3/3 | — |
| wiki/entities/open-source-humanoid-hardware.md | 3/3 | — |
| wiki/entities/opentac.md | 3/3 | — |
| wiki/entities/part-db.md | 3/3 | — |
| wiki/entities/pcb-motor.md | 3/3 | — |
| wiki/entities/qucs-s.md | 3/3 | — |
| wiki/entities/serial-studio.md | 3/3 | — |
| wiki/entities/simplefoc.md | 3/3 | — |
| wiki/entities/tinkercad.md | 3/3 | — |
| wiki/entities/tinymovr.md | 3/3 | — |
| wiki/entities/tscircuit.md | 3/3 | — |
| wiki/entities/ukmarsbot.md | 3/3 | — |
| wiki/entities/vesc.md | 3/3 | — |
| wiki/entities/wokwi.md | 3/3 | — |
| wiki/entities/wolfiemouse.md | 3/3 | — |
| wiki/overview/hub-actuator-drive-chain.md | 3/3 | — |
| wiki/overview/hub-communication.md | 3/3 | — |
| wiki/overview/humanoid-hardware-101-power-compute-electronics.md | 3/3 | — |
| wiki/overview/motor-drive-firmware-bus-protocols.md | 3/3 | — |
| wiki/queries/actuator-drive-chain-selection-loop.md | 3/3 | — |
| wiki/queries/field-robotics-troubleshooting.md | 3/3 | — |

## 原文与内容修正

- 通读 14 个官方仓库 README、KiCad 10.0 完整入门及 Autodesk Tinkercad 官方指南；所有归档列出准确阅读范围和版本。
- 通过 NIST 过程能力/抽样验收、ADI 串行接口/EMC、TI 器件、RLS/HEIDENHAIN 编码器、igus 运动电缆和 NVIDIA Jetson Thor 官方原文补齐剩余节点。
- 修正 Cpk 与良率关系、编码器分辨率与精度、屏蔽端接频段、电平转换与电隔离、模块算力及功耗口径；旧二手文章中的未核实数值明确保留来源边界。
- 微信文章和动态页面的访问限制仍如实保留；本次 `original` 指向实际读完的官方替代资料，不声称取回了受限文章全文。
- Konnect 旧 AGPL 仓与闭源后继、Tinymovr v3.0.0 公开快照与后续二进制分别说明；PCB Motor 参数区分 README 引用的论文与本仓实测。

## 分数的适用范围

本次维护现有节点的阅读骨架和文档溯源，没有新增项目实体或扩展源码运行时结论。已有源码图保留其核对提交与范围；新增功能流程图不冒充内部源码分析。3/3 不是所有外链、完整源码、付费标准、厂商宣传或实物实验均已复现的证明。

## 重跑

从 `roadmap/depth-pcb-design.md` 提取所有 `../wiki/*.md` 链接去重，作为参数传给 `python3 scripts/report_health.py`。实际 Stage 关系可用 `scripts/audit_depth_health.py` 的 `stage_wiki_links` 复核。
