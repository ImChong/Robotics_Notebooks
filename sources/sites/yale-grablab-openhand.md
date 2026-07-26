# Yale OpenHand Project（项目页）

- **标题：** Yale OpenHand Project — 开源快速原型机器人手系列
- **类型：** site / project-page
- **URL：** <https://www.eng.yale.edu/grablab/openhand/>
- **入库日期：** 2026-07-26
- **机构：** Yale University / Grab Lab（Aaron M. Dollar 组）
- **代码（CAD）：** <https://github.com/grablab/openhand-hardware>
- **控制软件：** <https://github.com/grablab/openhand_node>（活跃；旧仓 `openhand-software` 已弃用）
- **仿真：** <https://github.com/grablab/openhand_simulation>
- **许可：** Creative Commons Attribution-NonCommercial 3.0 Unported（CC BY-NC 3.0）
- **联系：** `v.patel[at]yale.edu`；邮件列表 openhands@mailman.yale.edu

## 一句话摘要

耶鲁 Grab Lab 的 **OpenHand** 项目页：聚合一系列 **3D 打印 + 腱驱动欠驱动** 开源手设计（Model T / T42 / O / Q / M2 / VF / Stewart / Sphinx / **F3** 等）、腕部耦合、HDM 制造说明与文献入口，目标是让硬件与软件研究可共进化，而非被迫用软件弥补封闭末端的机械短板。

## 公开信息要点（截至入库日）

### 项目动机（About）

- 商用灵巧手往往昂贵、平台绑定、难改型；研究者常只能在软件侧补偿固有机械劣势。
- OpenHand 意图建立可快速原型、可社区分叉的开源手族，使 **末端机械与算法研究可同迭代**。

### 设计范式（Design / Fabrication）

- **腱驱动欠驱动手指**：自适应贴合物体表面，降低对精细传感与复杂反馈的依赖。
- **Hybrid Deposition Manufacturing (HDM)**：3D 打印 + 树脂浇注，做多材料一体式柔顺关节与指垫。
- CAD 参数化：连杆长度、传动比、壳厚、孔位等可改并在相关零件间传播。

### 手型目录（HANDS，节选）

| 型号 | 要点（项目页 / README 归纳） |
|------|------------------------------|
| **Model T** | 四指、单执行器差分耦合；源自 SDM Hand；擅长自适应抓取，弱于手内操作 |
| **Model T42** | 双指双执行器；欠驱动自适应 + 部分手内原语 / 精密捏取 |
| **Model O** | 三指四执行器；类 Barrett / Robotiq / Reflex 拓扑；曾与 iHY / DARPA ARM 相关 |
| **Model Q** | 四执行器；精密指 + 可旋转强力指对，支持 finger-gaiting |
| **Model M2** | 单欠驱动指 + 可换拇指库；主动/拮抗腱可切换欠驱动与全驱动行为 |
| **Model VF** | T42 变体；指垫可变摩擦，利于平面手内操作 |
| **Stewart Hand** | Stewart–Gough 启发；面向 6-DoF 手内操作 |
| **Sphinx Hand** | 球面并联；约 3-DoF 空间旋转操作 |
| **Model F3** | T42 flexure–flexure 改编；腕相机形变估力（见独立页） |
| **Wrist Couplings** | 对接常见机械臂法兰的机械耦合 |

### 关键文献入口（页内列出）

- Ma, Odhner, Dollar — *Yale OpenHand Project…*，IEEE RAM 24(1), 2017
- Ma, Odhner, Dollar — *A Modular, Open-Source 3D Printed Underactuated Hand*，ICRA 2013
- Dollar & Howe — *The Highly Adaptive SDM Hand*，IJRR 2010
- Ma, Belter, Dollar — *Hybrid Deposition Manufacturing*，ASME JMR

## 源码 / 数据开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| **机械 CAD / STL** | **已开源** | [`grablab/openhand-hardware`](https://github.com/grablab/openhand-hardware)（各型号文件夹 + common/fingers/couplings） |
| **装配与制造文档** | **已开源** | 项目页各型号 Build 区 + 仓库内 PDF（如 Model F3 Assembly Guide 1.0） |
| **控制代码** | **已开源** | [`openhand_node`](https://github.com/grablab/openhand_node)（Model O / T / T42 + ROS；MIT）；旧 `openhand-software` 2019 起弃用 |
| **仿真** | **已开源** | [`openhand_simulation`](https://github.com/grablab/openhand_simulation) |
| **Model F3 视觉力估论文** | **审稿中** | 项目页写 `[paper under review]`；截至入库日未见公开 arXiv / DOI |
| **许可** | **CC BY-NC 3.0** | 非商业；学术使用需按仓库 LICENSE 引用 ICRA 2013 等 |

## 为何值得保留

- **欠驱动开源手的长期事实源**：相对一次性 demo 硬件，OpenHand 是可持续分叉、教学与抓取研究常用的 CAD 基准族。
- **与「高 DoF 仿人手」对照**：强调机械自适应与低执行器数，补全 [Allegro](../../wiki/entities/allegro-hand.md) / [RUKA-v2](../../wiki/entities/ruka-v2-hand.md) 另一侧选型。
- **Model F3 新方向**：腕相机形变估力、免 FT 力控擦拭 / peg / 书法，连接接触丰富操作与视觉力估计。

## 关联资料

- Model F3 型号页：[`sources/sites/yale-openhand-model-f3.md`](yale-openhand-model-f3.md)
- CAD 仓库：[`sources/repos/openhand-hardware.md`](../repos/openhand-hardware.md)
- 控制仓库：[`sources/repos/openhand_node.md`](../repos/openhand_node.md)
- Lab 站：<https://www.eng.yale.edu/grablab/>

## 对 wiki 的映射

- [Yale OpenHand](../../wiki/entities/yale-openhand.md)
