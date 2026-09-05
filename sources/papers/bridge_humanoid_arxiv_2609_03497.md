# BRIDGE（arXiv:2609.03497）

> 来源归档（ingest）

- **标题：** BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI
- **简称：** BRIDGE / Bridge
- **类型：** paper / humanoid / open-hardware / morphology-control
- **arXiv：** <https://arxiv.org/abs/2609.03497>
- **PDF：** <https://arxiv.org/pdf/2609.03497>
- **项目页：** <https://sites.google.com/view/bridgerobot> — 归档见 [`sources/sites/bridgerobot.md`](../sites/bridgerobot.md)
- **代码：** 截至 2026-09-05 **训练/部署代码未发布**（项目页写录用后再发）；通用 `github.com/bridge` **不是** 本平台仓
- **CAD：** 项目页提供整机 `.stp` 下载
- **机构：** 卡内基梅隆大学（CMU）、华中科技大学、JoyIn AI
- **入库日期：** 2026-09-04；**再核：** 2026-09-05
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 数据驱动的形态–控制共设计；落地 80 cm / 12.5 kg / 21 DoF / 约 1500 美元的 Bridge，对照 Bumi / K1 / ToddlerBot。

## 开源状态（步骤 2.5，2026-09-04；2026-09-05 再核）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（Google Sites）；真机视频 + CAD |
| 论文宣称 | open code / open design / co-design |
| CAD | **已开放** 整机 `.stp` |
| 其余 | 装配教程、电气、BOM、遥操作与高动态训练/部署代码：**录用后发布** |
| GitHub | 旧链指向 `github.com/bridge`（通用用户，**不是** 本平台仓） |

**结论：部分开源** — CAD 可下；控制与制造清单仍待发布。勿写成「硬件/控制已随仓发布」。

## 核心摘录

### 摘录 1：规格以 Table 1 为准

- Table 1 / 项目页：**0.8 m、12.5 kg、21 DoF、6 TFLOPS、~$1.5K**，open code / open design / co-design 三列打勾。
- Figure 1 图注写 88 cm / 13 kg — **与表不一致**，wiki 以表为准并注明。

**对 wiki 的映射：** [paper-bridge-humanoid](../../wiki/entities/paper-bridge-humanoid.md)

### 摘录 2：共设计四段 + 数字

- 腰部：运动学筛掉 pitch，动态跟踪选 **yaw-only**（\(\mathcal{E}_{\mathrm{dyn}}\) 0.02115 vs roll 0.02311）。
- Table 4：\(\mathcal{S}_{\mathrm{HL}}\) Bridge **0.5252**（Bumi 0.4321 / K1 0.4198 / ToddlerBot 0.3883）。
- Table 5（SONIC，LaFAN1+bones_seed）：SR **94.83**，MPJPE **0.0711**；高动态相对 K1 +4.70 点。
- SMPL 缩放均值 Bridge **1.021**，最接近人体比例。

**对 wiki 的映射：** [paper-bridge-humanoid](../../wiki/entities/paper-bridge-humanoid.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-04）
- [x] 2026-09-05 再核：CAD 已放、规格 80 cm、Table 4/5 写回 wiki
- [x] wiki 映射：`wiki/entities/paper-bridge-humanoid.md`
