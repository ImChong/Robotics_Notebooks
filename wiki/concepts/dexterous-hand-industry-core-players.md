---
type: concept
tags: [dexterous-hand, industry, humanoid-hardware, tactile, supply-chain, wechat]
status: complete
updated: 2026-11-03
related:
  - ../entities/zhaowei-zwhand-dexterous-hand.md
  - ../entities/inspire-robots-rh56-dexterous-hand.md
  - ../entities/linkerbot-linker-hand.md
  - ../concepts/dexterous-hand-payload-core-metrics.md
  - ../concepts/tactile-sensing.md
  - ../concepts/force-control-basics.md
  - ../concepts/impedance-control.md
  - ../overview/humanoid-hardware-101-sensing-end-effectors.md
  - ../overview/humanoid-hardware-101-supply-chain-economics.md
  - ../concepts/humanoid-mass-production-engineering.md
  - ../queries/contact-wrench-closed-loop.md
sources:
  - ../../sources/blogs/wechat_zane_hub_dexterous_hand_core_players_2026-11-03.md
summary: "依据 Zane Hub 综述梳理灵巧手五模块（传动 / 驱动 / 减速丝杠 / 感知 / 控制）、四类企业阵营与四个工程矛盾；12 家企业各有独立节点（企业数据为作者表述，未核实）。"
---

# 灵巧手行业核心企业与工程约束（Zane Hub 综述）

> 依据 [《目前灵巧手行业的核心企业，它们的优势体现在哪些方面？》](https://mp.weixin.qq.com/s/GgZdcerpXldZ0C_KvwFBoA)（Zane Hub，2026-11-03）整理。**企业与产品的参数、价格、产能均为作者表述，本库未核实**；文章结论是「表只能看位置，不能看胜负」。

## 一句话定义

**灵巧手是电机、减速器、传动件、传感器、驱动电路、控制算法与结构件共同决定的系统，任何一项短板都会把整手性能拉回最低水平；真正值得比较的，是企业把哪一段能力做成了可复制、可交付、可摊薄成本的东西。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| FOC | Field-Oriented Control | 磁场定向控制（驱动层） |
| η | Transmission efficiency | 传动效率，丝杠推力公式中的符号 |

## 为什么重要

- 文章称灵巧手约占人形整机成本的 14%–18%，其成本曲线决定人形机器人能否从展示品变成生产工具（作者表述）。
- 竞争重心从「谁有五指、谁自由度多」转向「一致性、寿命、成本能否同时控制」。

## 五个模块（作者归纳）

| 模块 | 要点 |
|------|------|
| 传动 | 腱绳（掌心空间大，但预紧 / 蠕变 / 迟滞，需重新标定）、连杆（刚性好，体积难压缩、易干涉）、丝杠（保持力好；滑动丝杠效率低 0.3–0.5，行星滚柱可达 0.85 以上但微型化贵）、直驱（无背隙、力反馈透明，但扭矩密度低）；近两年向「电机 + 减速器 + 丝杠」收敛，原因是平衡了刚度、保持能力、供应链与成本，而非理论最优 |
| 驱动 | 空心杯（低惯量、无齿槽，但功率密度低、堵转易温升、成本高）、无刷（效率高，需减速器与复杂驱动）、步进（成本低、保持好，高速扭矩下降）、微型伺服电缸（模块化、一致性好）；优势不在用了哪种电机，而在吃透并匹配结构与控制 |
| 减速与丝杠 | 微型化的门槛在制造而非设计：模数 0.2–0.3 的齿轮齿厚公差仅几微米，塑料收缩、热处理变形、粉末冶金密度均匀性、装配同轴度与润滑都会累积成回差 |
| 感知 | 压阻 / 电容 / 压电 / 光学 / 磁性；产品多是复合设计加温度补偿；评价要看阵列密度、量程、分辨率、采样率、迟滞、温漂、耐久、标定一致性，而不止「能不能测到 0.01 N」 |
| 控制 | 位置 → 速度 → 力矩 → 阻抗 → 力位混合；同样的电机与传感器，手内操作水平可差一个代际，差在参数辨识、摩擦补偿与触觉闭环（见 [阻抗控制](./impedance-control.md)、[力控基础](./force-control-basics.md)） |

## 四个工程矛盾

1. **自由度与重量**：每增一个主动自由度就要加电机、减速器、轴承、走线与驱动通道，重量与成本近似线性、可靠性超线性变差；应看「有效自由度」。
2. **刚度与柔顺**：高刚度精确但碰撞易损，柔顺安全但带宽受限，靠阻抗 / 力位混合控制兼顾。丝杠推力简化式 F = 2π·T·η / P；作者算例 T=0.05 N·m、导程 1 mm、η=0.5 → 约 157 N。
3. **性能与可制造性**：42 自由度可作展示，产线要考虑装配节拍、来料一致性、测试工位与售后；可制造性是把性能做成可重复的过程能力。
4. **单价与寿命**：6 自由度工业手可做到万元级，17–20 自由度要数万元；整手寿命遵循木桶效应，最先失效的零件决定维护周期。

## 企业阵营与节点（12 家）

| 企业 / 产品 | 阵营 | 节点 | 状态 |
|-------------|------|------|------|
| 帕西尼 PaXini（DexH13 GEN2） | 感知起家 | [paxini](../entities/paxini.md) | 复用 |
| 星动纪元 Robotera（XHAND1） | 整机自研 | [robotera](../entities/robotera.md) | 复用 |
| Shadow Robot（Shadow Hand） | 海外科研标杆 | [shadow-hand](../entities/shadow-hand.md) | 复用 |
| 特斯拉 Optimus 手部 | 整机自研 / 海外 | [tesla-optimus](../entities/tesla-optimus.md) | 复用 |
| 兆威机电 ZWHAND 灵巧手 | 零部件下探（微型传动） | [zhaowei-zwhand-dexterous-hand](../entities/zhaowei-zwhand-dexterous-hand.md) | 新建 |
| 因时机器人 RH56 灵巧手 | 零部件下探（伺服电缸） | [inspire-robots-rh56-dexterous-hand](../entities/inspire-robots-rh56-dexterous-hand.md) | 新建 |
| 灵心巧手 Linker Hand | 灵巧手专营 | [linkerbot-linker-hand](../entities/linkerbot-linker-hand.md) | 新建 |
| 智元 SkillHand | 本体自研 | [agibot-skillhand](../entities/agibot-skillhand.md) | 新建 |
| 傲意科技 OHand 系列 | 假肢背景 | [oymotion-ohand](../entities/oymotion-ohand.md) | 新建 |
| 途见科技 可拉伸电子皮肤 | 感知专精（触觉器件） | [tujian-electronic-skin](../entities/tujian-electronic-skin.md) | 新建 |
| 他山科技 电容式触觉芯片 | 感知专精（触觉器件） | [tashan-capacitive-tactile](../entities/tashan-capacitive-tactile.md) | 新建 |
| 汉威科技 柔性传感 | 感知专精（触觉器件） | [hanwei-flexible-sensing](../entities/hanwei-flexible-sensing.md) | 新建 |

### 作者对阵营的判断

- **零部件下探**（兆威、因时）：优势在制造与成本，短板在场景与算法。
- **本体自研**（星动纪元、智元、特斯拉）：优势在整机协同与数据，短板在规模与成本。
- **感知专精**（途见、他山、汉威）：优势在材料 / 工艺 / 标定，短板在系统集成；壁垒在「传感器 + 算法 + 数据接口」组合而非单一敏感材料。
- **海外标杆**：Shadow Hand 把灵巧做到极致也把成本推到极致（文章称价格 100–220 万元）；特斯拉 Optimus 手执行器集中在前臂，目标自由度 22、一度传出收敛到 16，说明方案仍未收敛。
- 未来分工（作者预判）：传动与结构由零部件企业供给、触觉由感知企业供给、整机厂做定义与集成。

## 选型三标准（作者）

1. 看是否有真实批量交付记录，而非发布会与演示视频；
2. 看关键指标的测试条件（寿命、温升、一致性）是否可复现；
3. 看接口是否开放、是否支持二次开发与长期维护。

## 关联页面

- [灵巧手负载核心指标](./dexterous-hand-payload-core-metrics.md)
- [触觉感知](./tactile-sensing.md)
- [力控基础](./force-control-basics.md)
- [阻抗控制](./impedance-control.md)
- [人形硬件 101：感知与末端执行器](../overview/humanoid-hardware-101-sensing-end-effectors.md)
- [人形硬件 101：供应链与经济](../overview/humanoid-hardware-101-supply-chain-economics.md)
- [人形量产工程能力](./humanoid-mass-production-engineering.md)
- [接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 本页是该链路的硬件供给侧：触觉器件对应 ① 感知/估计层、传动背隙与腱绳蠕变影响 ② 表示层的标定、阻抗/力位混合对应 ③ 控制层，防止把「手上装了触觉传感器」误当成已具备接触力闭环（文章指出同样的电机与传感器，差距在参数辨识、摩擦补偿与触觉闭环）。

## 参考来源

- [wechat_zane_hub_dexterous_hand_core_players_2026-11-03.md](../../sources/blogs/wechat_zane_hub_dexterous_hand_core_players_2026-11-03.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/GgZdcerpXldZ0C_KvwFBoA)
