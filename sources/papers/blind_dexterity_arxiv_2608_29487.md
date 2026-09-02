# Blind Dexterity（arXiv:2608.29487）

> 来源归档（ingest）

- **标题：** Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception
- **类型：** paper / humanoid / manipulation / proprioception / interactive-perception / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2608.29487>
- **PDF：** <https://arxiv.org/pdf/2608.29487>
- **HTML：** <https://arxiv.org/html/2608.29487>
- **项目页：** <https://aditya.bhatts.org/BlindDexterity/>
- **机构：** 达姆施塔特工业大学（TU Darmstadt）IAS Lab；德国人工智能研究中心（DFKI）；hessian.AI
- **作者：** Aditya Bhatt、Oleg Kaidanov、Puze Liu、Jan Peters
- **发表 / 上传：** 2026-08（arXiv）
- **平台：** Unitree G1
- **入库日期：** 2026-09-02

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.29487](https://arxiv.org/abs/2608.29487) | 论文与补充视频 |
| 项目页 | [BlindDexterity](https://aditya.bhatts.org/BlindDexterity/) | 四类任务视频与核心论点 |
| 平台 | [Unitree G1](../../wiki/entities/unitree-g1.md) | 29-DoF 实验人形 |
| 对照 | [GentleHumanoid](../../wiki/entities/paper-gentlehumanoid.md) | 柔顺全身 RL（本文把柔顺当传感通道） |

## 开源状态（步骤 2.5，2026-09-02 复核）

- **宣称将开源 / 待发布：** 项目页 **Code (to be released)**；截至入库日 **无** GitHub URL。
- **论文：** 「Complete task configurations … will be released with the code.」
- **处理：** wiki 标待发布；`## 源码运行时序图` 标不适用。

## 摘要级要点

- **问题：** 人形操作常依赖相机/F/T/触觉；能否仅用 onboard 本体（编码器 + 可选 IMU）做全身灵巧操作？
- **洞察：** 柔顺 PD 下命令-测量关节残差 ∝ 力矩；短历史 + 上一动作 `a_{t-1}` 形成粗粒度「全身触觉」；策略主动接触以解码物体位姿。
- **四类任务：** (1) 无 IMU 推抗行走；(2) 足球主动搜索停球；(3) 滑板登板；(4) 手提箱提柄（含 VS 可变刚度）。
- **方法：** Isaac Sim + PPO；MLP 策略 + 独立 MLP 状态估计器（诊断/消融）；4096 envs；50 Hz。
- **关键结论：** Blind-from-scratch **优于** DAgger 蒸馏学生；+SE 估计器反馈无一致增益；真机定性验证 sim 定量。

## 核心摘录（面向 wiki 编译）

### 1) 隐式接触信号（§IV-A）

\(e_t = q^{des}_{t-1} - q_t\)；PD 下 \(\tau \approx K_p e_t - K_d \dot{q}_t\)。策略观测含 \(a_{t-1}\) 即可恢复残差动态。

### 2) 行走消融（Table I，5 seeds）

| 配置 | Survival (%) | Lin Vel Err (m/s) |
|------|-------------|-------------------|
| +IMU+PE | 96.1±0.5 | 0.3370±0.0040 |
| -IMU+PE | 90.4±1.1 | 0.3867±0.0047 |
| -IMU-PE | 89.1±1.1 | 0.4408±0.0150 |

### 3) 足球任务（Table II，best seed）

| Policy | Success (%) | Loc Error (cm) |
|--------|------------|----------------|
| Privileged | 99.9 | — |
| Distilled | 67.7 | — |
| Blind-SE | 94.3 | 8.97 |
| Blind+SE | 99.3 | 8.09 |

蒸馏学生仅把脚放在 spawn 分布中心；blind 策略学习 sweep + ankle wiggle 主动搜索。

### 4) 训练量级

- 行走/足球：5000 PPO iter；手提箱：6000；滑板：8000。
- 历史长度：操作任务 K=5（0.1 s）；行走 K=1。

## 对 wiki 的映射

- 沉淀实体页：[Blind Dexterity](../../wiki/entities/paper-blind-dexterity.md)
- 交叉补强：[unitree-g1](../../wiki/entities/unitree-g1.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[manipulation](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] arXiv HTML 方法 / Table I–III 摘录
- [x] 项目页开源核查：Code to be released
- [x] 升格 `wiki/entities/paper-blind-dexterity.md`
