---
type: entity
tags: [paper, stanford, realab, manipulation]
status: complete
updated: 2026-09-15
arxiv: "2505.24853"
venue: "ICML 2026"
code: https://project-dexmachina.github.io/
related:
  - ./paper-chord-contact-wrench-dexterous-manipulation.md
  - ./paper-demomimic.md
  - ../methods/regrind-retargeting-guided-rl.md
  - ../tasks/bimanual-manipulation.md
  - ../overview/realab-14-papers-technology-map-2026.md
sources:
  - ../../sources/papers/dexmachina_arxiv_2505_24853.md
  - ../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md
summary: "DexMachina（ICML 2026）：VOC 课程+任务/运动/接触奖励的功能重定向；双手灵巧长时程 benchmark；仿真显著优于基线；真机鲁棒性待验。"
---

# DexMachina（arXiv:2505.24853）

**DexMachina**（Mandi Zhao, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song；Stanford University; NVIDIA；[arXiv:2505.24853](https://arxiv.org/abs/2505.24853)，[项目页](https://project-dexmachina.github.io/)）— 从人类手–物示范学习机器人双手灵巧策略：虚拟物体控制器（VOC）课程 + 多奖励 RL，在仿真 benchmark 上做功能重定向。

## 一句话定义

从人类手–物示范学习机器人双手灵巧策略：虚拟物体控制器（VOC）课程 + 多奖励 RL，在仿真 benchmark 上做功能重定向。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VOC | Virtual Object Controller | 强度衰减的物体辅助控制 |
| RL | Reinforcement Learning | 策略学习范式 |
| DM | DexMachina | 本文算法与 benchmark 名 |
| IK | Inverse Kinematics | 对比的功能重定向基线 |

## 为什么重要

双手灵巧操作动作空间大、时空不连续、人机手形态差大，直接模仿易失败。

## 流程总览

```mermaid
flowchart TD
    DEMO["人类手-物示范 D<br/>密集跟踪的物体状态 G 与手部姿态 H<br/>关节物体：部件位姿 + 旋转关节角"] --> KR
    subgraph PRE["示范预处理"]
        KR["仅运动学重定向<br/>匹配灵巧手与人手姿态"] --> REF["仿真回放（物体固定、软控制目标）<br/>得到碰撞感知的参考关节 Q 与参考关键点 X"]
        KR --> CON["基于距离近似手-物接触<br/>得到接触位置 C 与有效掩码 M"]
    end
    REF --> ACT
    subgraph RL["RL 环境（PPO，Genesis 仿真，双手同时控制）"]
        ACT["混合动作<br/>6-DoF 手腕：参考关节作基础动作 + 策略残差<br/>手指：按关节限位归一化的绝对动作"]
        RW["奖励加权和<br/>任务奖励（位置×旋转×关节角，乘积形式）<br/>运动模仿 r_imi + 行为克隆 r_bc + 接触奖励 r_con"]
    end
    REF --> RW
    CON --> RW
    subgraph CUR["VOC 自动课程"]
        VOC["虚拟物体控制器<br/>PD 弹簧-阻尼驱动物体沿示范轨迹运动<br/>（仿真特权信息）"] --> DEC["按近期奖励均值达到阈值<br/>指数衰减增益 kp、kv，直至为 0"]
    end
    ACT --> POL["策略 π（RL 训练）"]
    RW --> POL
    VOC -. 早期强辅助，后期撤除 .-> POL
    POL --> DEC
    POL --> OUT["无 VOC 下的双手灵巧策略<br/>跟踪示范的物体轨迹"]
```

图依据：论文 §4.1–4.3 / Fig. 2 / Algorithm 1

## 核心原理（方法）

从示范提取任务/运动/接触奖励；VOC 先驱动物体到目标再让策略接管；多灵巧手多任务 benchmark。

## 实验与评测

仿真多任务显著优于 IK/位置引导等基线；支持跨硬件功能比较。

## 结论

DexMachina 把 VOC 课程做成可复现的灵巧功能重定向平台，但真机过渡仍是开放问题。

- VOC 降低早期探索难度
- benchmark 覆盖多灵巧手与关节物体
- 可比较硬件设计的功能差异
- CHORD 等后续工作将其作位置引导基线
- 真机 VOC 平滑过渡鲁棒性未充分验证

## 源码运行时序图

**不适用**（截至 2026-08-18：无统一公开可运行代码仓库，或本文为综述/控制器论文以项目页演示为主）。

## 局限与风险

主要在仿真验证；接触奖励设计敏感；与 interaction mesh 方法（REGRIND）路线不同。

## 与其他工作对比

相对 CHORD 的 wrench 空间奖励，DexMachina 偏位置+VOC；相对 REGRIND，缺交互 mesh 语义。

## 关联页面

- [paper-chord-contact-wrench-dexterous-manipulation](./paper-chord-contact-wrench-dexterous-manipulation.md)
- [DemoMimic](./paper-demomimic.md) — 项目页将 DexMachina* 作 sim 高、真机低的对照基线（接触中心 AR/SCR）
- [regrind-retargeting-guided-rl](../methods/regrind-retargeting-guided-rl.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [REALab 14 篇技术地图](../overview/realab-14-papers-technology-map-2026.md)

## 参考来源

- [dexmachina_arxiv_2505_24853.md](../../sources/papers/dexmachina_arxiv_2505_24853.md)
- [wechat_shenlan_realab_14_papers_2026.md](../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md)
- [arXiv PDF](https://arxiv.org/pdf/2505.24853)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 功能重定向问题定义、§4 方法（§4.1 RL 环境与任务奖励、§4.2 动作与辅助奖励、§4.3 虚拟物体控制器自动课程）；§5 实验仅略读，附录未读


## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2505.24853>
- 项目页：<https://project-dexmachina.github.io/>
