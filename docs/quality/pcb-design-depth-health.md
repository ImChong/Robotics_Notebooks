# PCB 设计纵深路线节点健康度复核

- 复核日期：2026-11-16。
- 范围：路线页所有直接 wiki 链接，含页尾导航；41 个唯一节点。实际 Stage 内 36 个节点，其中 28 个满分。
- 沿用 `generate_link_graph.compute_health`，未修改评分或过滤节点。
- 全部直接关联节点：满分由 6/41 提升至 32/41；剩余 9 页为 2/3。路线页本身为 3/3。
- 官方 README 通读、KiCad 入门全文、既有原文归档与章节骨架分别核对；这些不是硬件复现或全部源码审查的证明。

## 实测结果

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/concepts/can-fd.md | 3/3 | — |
| wiki/concepts/ethercat-protocol.md | 3/3 | — |
| wiki/concepts/field-oriented-control.md | 3/3 | — |
| wiki/concepts/humanoid-mass-production-engineering.md | 2/3 | 原文未通读 |
| wiki/concepts/humanoid-testing-workflow.md | 3/3 | — |
| wiki/concepts/humanoid-wire-harness-types-protection.md | 2/3 | 原文未通读 |
| wiki/concepts/joint-encoder-selection.md | 2/3 | 原文未通读 |
| wiki/concepts/joint-module-electromagnetic-isolation.md | 2/3 | 原文未通读 |
| wiki/concepts/micromouse.md | 3/3 | — |
| wiki/concepts/robot-onboard-communication-architecture.md | 3/3 | — |
| wiki/concepts/robot-power-distribution-architecture.md | 3/3 | — |
| wiki/concepts/rs-232-serial-interface.md | 2/3 | 原文未通读 |
| wiki/concepts/rs-485-serial-bus.md | 3/3 | — |
| wiki/concepts/ttl-serial-logic-level.md | 2/3 | 原文未通读 |
| wiki/concepts/uart-serial-communication.md | 2/3 | 原文未通读 |
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
| wiki/entities/tinkercad.md | 2/3 | 原文未通读 |
| wiki/entities/tinymovr.md | 3/3 | — |
| wiki/entities/tscircuit.md | 3/3 | — |
| wiki/entities/ukmarsbot.md | 3/3 | — |
| wiki/entities/vesc.md | 3/3 | — |
| wiki/entities/wokwi.md | 3/3 | — |
| wiki/entities/wolfiemouse.md | 3/3 | — |
| wiki/overview/hub-actuator-drive-chain.md | 3/3 | — |
| wiki/overview/hub-communication.md | 3/3 | — |
| wiki/overview/humanoid-hardware-101-power-compute-electronics.md | 2/3 | 原文未通读 |
| wiki/overview/motor-drive-firmware-bus-protocols.md | 3/3 | — |
| wiki/queries/actuator-drive-chain-selection-loop.md | 3/3 | — |
| wiki/queries/field-robotics-troubleshooting.md | 3/3 | — |

## 待补原文与访问阻塞

- [humanoid-mass-production-engineering](../../wiki/concepts/humanoid-mass-production-engineering.md)：原文章浏览器访问失败（ERR_TUNNEL_CONNECTION_FAILED），无完整原文快照，未标通读。
- [humanoid-wire-harness-types-protection](../../wiki/concepts/humanoid-wire-harness-types-protection.md)：原文章浏览器访问失败（ERR_TUNNEL_CONNECTION_FAILED），保留作者数值未经核实的限定。
- [joint-encoder-selection](../../wiki/concepts/joint-encoder-selection.md)：原文章浏览器访问失败（ERR_TUNNEL_CONNECTION_FAILED），未以现有摘要替代全文。
- [joint-module-electromagnetic-isolation](../../wiki/concepts/joint-module-electromagnetic-isolation.md)：原文章浏览器访问失败（ERR_TUNNEL_CONNECTION_FAILED），未以现有摘要替代全文。
- [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)：现有来源为标准/教程索引；补充教程返回 HTTP 403，尚未取得并通读匹配的主要原文。
- [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)：现有来源为合集索引；Microsoft 原链接返回 404，补充逻辑电平教程返回 HTTP 403，未冒充读完标准或数据手册。
- [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)：Microsoft 原链接返回 404，补充串行通信教程返回 HTTP 403，原始教程尚未完整通读。
- [tinkercad](../../wiki/entities/tinkercad.md)：HTTP 仅取得页面标题，浏览器导航失败；未将动态教程空壳记为通读。
- [humanoid-hardware-101-power-compute-electronics](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md)：现有主要来源是二手综合文章；未完成其电池/计算/PCB 各项官方技术原文核对，不将综述登记为 original。

## 源码与复现边界

- 保留已有源码时序图及其核对提交；本次新增流程图描述官方 README 的功能数据流，不能代替新的源码运行分析。
- moteus、VESC、Tinymovr、UKMARSBOT、WolfieMouse、tscircuit、Serial Studio、SimpleFOC、KiCad 尚需补充或扩展实际源码入口核对；因此即使机器健康度为 3/3，也不宣称全部 ingest 要求已完成。
- Konnect 原 AGPL 仓转向归档，后续 Konnect-EDA 为闭源产品；Tinymovr 公开源码为 v3.0.0 快照，后续版本仅专有二进制。
- PCB Motor 的 20 极/六层等数字来自 README 引用的论文，未验证为本仓成品参数。

## 重跑方式

从路线页提取全部 `../wiki/*.md` 链接去重，作为参数传给 `python3 scripts/report_health.py`；仅看实际 Stage 挂载可复用 `scripts/audit_depth_health.py` 的 `stage_wiki_links`。
