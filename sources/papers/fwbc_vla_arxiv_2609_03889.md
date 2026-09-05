# FWBC-VLA（arXiv:2609.03889）

> 来源归档（ingest）

- **标题：** FWBC-VLA: Force-Aware Whole-Body Compensation for Contact-Rich Loco-Manipulation
- **简称：** FWBC-VLA
- **类型：** paper / vla / wbc / loco-manipulation / sensorless-force
- **arXiv：** <https://arxiv.org/abs/2609.03889>
- **PDF：** <https://arxiv.org/pdf/2609.03889>
- **HTML：** <https://arxiv.org/html/2609.03889>
- **项目页：** 截至 2026-09-05 **未见** 独立项目页（用户与论文均未给 URL）
- **代码：** 截至 2026-09-05 **未见** 官方仓；论文未承诺 GitHub
- **数据：** WL&Arm Dataset（>5000 条）论文写 **will be released publicly**，入库日无 URL
- **机构：** 浙江大学、上海人工智能实验室、清华大学、北京中关村学院、云深处科技、浙江科技学院
- **作者：** Yutian Zhang\*、Siyuan Ma\*、Liwen Yang、Yang Li、Ce Hao、Haozhen Chi、Dong Wei\*、Qiaojun Yu\*、Dibo Hou\*（\* 同等贡献 / 通讯）
- **入库日期：** 2026-09-05
- **一句话说明：** 无 F/T 传感器的接触感知接口：HSR-Force 从本体力矩估残差，同时条件化 π₀.₅ 动作专家并生成有界底盘补偿，再交给轮足 WBC。

## 开源状态（步骤 2.5，2026-09-05）

| 组件 | 状态 |
|------|------|
| 项目页 | **无**（论文/用户均未列 `*.github.io` 或 lab 页） |
| GitHub / HF | **未见** 官方仓或权重 |
| 数据集 | **宣称将开源**（WL&Arm >5000 episodes，无 URL） |
| 论文承诺 | 未写 "code will be released"；只写数据集将公开 |

**结论：确认未开源（数据宣称待发布）。** 勿建 `sources/repos/` / `sources/sites/`。源码运行时序图写「不适用」。勿把「无传感器」写成「无复现材料」。

## 核心摘录

### 摘录 1：双回路，而不是只给臂加力

- 现有力觉 VLA（ForceVLA / ACP）多是 **臂中心**，且依赖专用 F/T。
- 轮足+臂上，末端力会沿臂传到机身；WBC 只能稳，**分不清任务力与扰动**。
- FWBC-VLA 用同一套无传感器交互表示喂两条回路：VLA 动作专家 + 底盘补偿生成器。

**对 wiki 的映射：** [paper-fwbc-vla](../../wiki/entities/paper-fwbc-vla.md)

### 摘录 2：HSR-Force（200 Hz 双 LSTM + 固定门）

- 把测量力矩拆成自由运动 + 外部残差 + 未建模项；估的是 **关节残差**，不是标定 wrench。
- History expert 用因果力矩/运动历史压自由运动噪声；State expert 用当前臂/基/腿/IMU、**不读力矩历史**，避免把持续接触吸进自由运动。
- 固定门 \(\alpha_t=G(\|r_{\mathrm{state}}\|_2)\)：自由运动偏 history，接触偏 state。
- 接触描述子 \(d^{\mathrm{int}}=[s_t,\Delta s_t]\)（强度 + 增减趋势）；\(K=13\)（200 Hz 上约 60 ms）编成 force token，**只 late-fuse 进 action expert**。
- 残差经阻尼最小二乘 Jacobian 投到 EE / 机身坐标系，供补偿支路。

**对 wiki 的映射：** [paper-fwbc-vla](../../wiki/entities/paper-fwbc-vla.md)、[contact-estimation](../../wiki/concepts/contact-estimation.md)

### 摘录 3：WL&Arm 与真机数字

- Pico 混合位/力遥操作；力意图只作元数据，**不进** 估计器 / VLA / 补偿。
- VLA 输出基座速度 + 转向角，而不是轮腿关节，以降低与预训练本体错配。
- 组成：拾放瓶 41%、擦白板 25%、开门 21%；标定载荷 0.36 / 0.72 kg（3.53 / 7.06 N）。
- HSR-Force：零载 Force MAE **0.15 N**，Touch AUC **0.97**，Door-phase AUC **0.85**（对照 NEXT / GMO-SI / DF-MLP）。
- 云深处 M20S + CM1 6-DoF 臂；门闭门器手柄约 **50 N**。
- 擦白板终段成功率 **64%**（ForceVLA 24%、OpenPI 0.5 12%）；开门终段 **52%**（ForceVLA 12%、OpenPI 0.5 0%）。
- 消融表：无力 12.0 → 仅力接口 35.0 → 加底盘补偿 **65**（正文另写 59.5，以表为准）。

**对 wiki 的映射：** [paper-fwbc-vla](../../wiki/entities/paper-fwbc-vla.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[vla](../../wiki/methods/vla.md)

## 当前提炼状态

- [x] arXiv HTML 全文核对（2026-09-05）
- [x] 步骤 2.5：无项目页、无仓、数据宣称待发布
- [x] wiki 映射：`wiki/entities/paper-fwbc-vla.md`
