---
type: entity
tags: [hardware, humanoid, industry, actuator, manipulation, dexterous-hand]
status: complete
updated: 2026-05-12
related:
  - ./humanoid-robot.md
  - ./allegro-hand.md
  - ./shadow-hand.md
  - ../tasks/manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../concepts/armature-modeling.md
  - ../overview/notable-commercial-robot-platforms.md
  - ../methods/vla.md
sources:
  - ../../sources/sites/wuji_robotics.md
summary: "舞肌科技（上海舞肌）面向具身机器人提供关节级电机（F 系列 / Pan Motor 叙事）与官方文档确认的五指灵巧手 Wuji Hand，配套 SDK、ROS2、MuJoCo 描述与遥操作 Retargeting；研发在上海、量产在常州。"
---

# 舞肌科技（上海舞肌科技有限公司）

## 一句话定义

**舞肌科技** 面向 **具身 AI 机器人** 提供两类常被并列讨论的硬件叙事：**关节级电机方案**（**F 系列** 内转子永磁无刷、「**Pan Motor**」品牌报道）与 **五指灵巧手整机**（官方产品名 **Wuji Hand**）。电机侧公开话术强调 **扭矩密度**（常见引用 **约 200 g 本体、8 N·m 峰值扭矩**）；灵巧手侧以 **docs.wuji.tech** 上的规格表与软件栈为准。研发与运营主体在上海，制造在江苏常州。

## 为什么重要

- **供应链视角**：人形与灵巧操作爆发带动 **小型高扭矩关节电机** 与 **高主动 DoF 灵巧手** 需求；舞肌在中文语境中同时出现在 **动力部件** 与 **灵巧手盘点** 两类材料里，适合与整机厂（如 [Unitree](./unitree.md)、[Figure AI](./figure-ai.md)）及科研常用灵巧手平台（如 [Allegro Hand](./allegro-hand.md)、[Shadow Hand](./shadow-hand.md)）对照阅读。
- **仿真与建模衔接**：电机扭矩密度进入 **反射惯量、热与连续扭矩包络**（见 [Armature 建模](../concepts/armature-modeling.md)）；灵巧手侧官方提供 **URDF/MJCF、MuJoCo 示例仓库** 与 **ROS2** 入口，便于与 [Manipulation](../tasks/manipulation.md) 管线、手内多接触 [灵巧手运动学](../concepts/dexterous-kinematics.md) 对齐。
- **信息溯源训练**：融资与部分产业稿分散在 **招聘平台、媒体转载**；**灵巧手参数应优先采信官方文档中心**，避免直接抄「盘点类」二手表格。

## 公开产品与组织叙事（归纳）

| 维度 | 公开表述（归纳） | 备注 |
|------|------------------|------|
| **法律主体** | 上海舞肌科技有限公司；英文常用 *Shanghai Wuji Technology* | 工商与招聘页一致 |
| **F 系列电机** | 2022 年推出首款；内转子永磁无刷路线上的结构重构叙事 | 技术细节需 datasheet |
| **Pan Motor** | 部分 2025 年媒体报道中的品牌称呼，与 F 系列并提 | 见网易转载稿 |
| **Wuji Hand** | 官方文档定义的 **20 主动自由度** 仿生灵巧手，面向科研、机器人集成与人机交互 | 见 [Wuji Hand 一节](#wuji-hand五指灵巧手) |
| **地理** | 研发：上海虹桥商务区；生产：江苏常州；注册地址在嘉定江桥（招聘页） | 以合同与官方联系为准 |
| **融资** | 公开招聘文案称累计约 **4000 万元**；媒体报道 **B 轮** 及多家机构名字 | 投资条款以工商披露为准 |

## Wuji Hand（五指灵巧手）

### 官方定位与生态（归纳）

根据 **舞肌科技文档中心**（`docs.wuji.tech`）当前公开结构，**Wuji Hand** 被描述为舞肌自研的 **20 主动自由度** 仿生灵巧手，应用场景包括 **科研实验、机器人集成、人机交互**。文档树同时指向 **Python/C++ SDK**、**ROS2 驱动**、**HMI 上位机**、**Retargeting**（文档中出现 **Apple Vision Pro** 相关配置说明）、以及 **URDF/MJCF 描述包与 MuJoCo 可视化**（并引用如 [`wuji-technology/mujoco-sim`](https://github.com/wuji-technology/mujoco-sim) 等 GitHub 仓库入口）。以上模块边界与版本以官方文档为准。

### 核心参数（摘自官方「产品介绍」参数表，工程以页面为准）

| 类别 | 项目 | 典型值或说明 |
|------|------|----------------|
| 结构 | 主动自由度 | **20**（官方表：每指 4 个） |
| 结构 | 抓握构型 | 可全对指；四指支持侧摆 |
| 结构 | 驱动方式 | 自锁旋转直驱关节；FOC 矢量控制 |
| 几何 | 最大抓握直径 | **100 mm** |
| 几何 | 自重 | **580 ± 10 g**（不含线缆） |
| 几何 | 外形尺寸 | **201 mm × 75 mm × 50 mm** |
| 负载 | 指尖力 | **15 N** |
| 负载 | 整手抓握最大静载 | **10 kg**（文档表中与「整手钩握最大静载」同列数值） |
| 控制 | 控制频率 | **1000 Hz × 20 轴** |
| 接口 | 通信 | **USB 2.0**（硬件上文档配图说明为 **USB Type-C** 连接上位机） |
| 电气 | 工作电压 | **12–20 V DC**（文档提及标配 12 V 20 A 适配器） |

### 产业报道中的补充线索（非官方规格书）

- **NE 时代** 产业链盘点稿在「舞肌科技」小节中另提到 **压阻方案触觉手套** 及多种掌内操作演示，并给出 **约 5 万元级** 定价等说法；该文属于 **媒体归纳**，与官方文档在 **自重（稿写约 600 g）** 等细节上不一致，**不宜单独作为验收依据**。

## 常见误区或局限

- **「200 g / 8 N·m」≠ 可直接写进仿真 `motor` 块**：通常是 **电机本体质段** 的峰值指标，不含减速器、法兰、走线与散热边界；与 **连续扭矩、温升、母线电压、控制频率** 联合才有工程意义。
- **媒体报道 ≠ 规格书**：转载稿可能含 **AI 生成说明**（例如网易该文脚注），引用融资与参数时应 **回链工商或官方新闻稿**。
- **与「伏为电机」等 DD 马达厂商勿混淆**：中文「直驱 / 力矩电机」检索常混入 **工业 DD 马达** 供应商（域名、主体均不同），选型时应对照 **营业执照与商标**。
- **灵巧手「盘点文章」≠ 官方 datasheet**：链长、背隙、温升与寿命等 **应以 docs.wuji.tech 与随货文件为准**。

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [灵巧手运动学](../concepts/dexterous-kinematics.md)
- [Allegro Hand](./allegro-hand.md)
- [Shadow Hand](./shadow-hand.md)
- [Armature 建模](../concepts/armature-modeling.md)
- [市面知名机器人平台纵览](../overview/notable-commercial-robot-platforms.md)
- [VLA](../methods/vla.md) — 招聘与产业报道中偶见「多模态 / VLA」方向岗位，与整机 **感知–语言–动作** 栈相关，但 **不等于** 该公司已发布可复现的公开模型栈

## 参考来源

- [舞肌科技原始资料汇编](../../sources/sites/wuji_robotics.md)

## 推荐继续阅读

- [舞肌文档中心 · Wuji Hand（中文总入口）](https://docs.wuji.tech/docs/zh/wuji-hand/latest/)
- [Wuji Hand 产品介绍（中文）](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview/)
- [国家大学生就业服务平台 · 舞肌企业简介](https://yazjy.ncss.cn/student/jobs/PkW5wLBq1LsLjuuFFFZhch/corp.html) — 公司简介载体（偏电机叙事）
- [网易 · 舞肌科技 B 轮融资转载](https://www.163.com/dy/article/KELVSCVC051984TV.html) — 含「Pan Motor」叙事与融资线索（注意 AI 生成声明）
