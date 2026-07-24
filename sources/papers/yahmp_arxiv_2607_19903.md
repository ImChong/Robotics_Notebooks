# What Matters in Humanoid General Motion Tracking? An Empirical Study（YAHMP，arXiv:2607.19903）

> 来源归档（ingest）

- **标题：** What Matters in Humanoid General Motion Tracking? An Empirical Study
- **缩写 / 框架：** **YAHMP**（Yet Another Humanoid Motion tracking Policy）
- **类型：** paper / humanoid / motion-tracking / empirical-study / sim2real / onnx-deploy
- **arXiv：** <https://arxiv.org/abs/2607.19903>（Submitted 2026-07-22；PDF：<https://arxiv.org/pdf/2607.19903>）
- **代码：** <https://github.com/fabio-amadio/yahmp>（Apache-2.0；论文亦写 `hucebot/yahmp`，为同仓库 org fork）— 归档见 [`sources/repos/yahmp.md`](../repos/yahmp.md)
- **补充视频：** <https://youtu.be/BH6FpQzwm8M>
- **作者：** Fabio Amadio、Enrico Mingo Hoffman
- **机构：** Inria / Université de Lorraine / CNRS（Nancy, France）；ANR MeRLin（ANR-24-CE33-0753-01）资助
- **入库日期：** 2026-07-24
- **一句话说明：** 在 Unitree G1 上用开源模块化框架 **YAHMP**（基于 mjlab）做受控消融：命令表示、观测历史、动作表示、驱动剖面、手部力随机化、Teacher–Student；与同数据重训的 TWIST2 对照，并 zero-shot ONNX/真机部署。

## 开源状态（步骤 2.5）

- **仓库核查（2026-07-24）：** [fabio-amadio/yahmp](https://github.com/fabio-amadio/yahmp) 公开：`uv sync` 训练、`export_checkpoint_to_onnx`、`run_yahmp_onnx_mujoco`、预置 `assets/models/g1_yahmp.onnx` 与 TWIST2 ONNX 对照脚本；运动资产需自下 Google Drive（OMOMO+AMASS 重定向包）。
- **结论：** **已开源**（训练 / 评测 / ONNX 部署入口齐全）。

## 摘录 1：问题与主张（§I）

- **痛点：** 近年全身 GMT 管线设计选择多，但消融常绑死单一方法；跨方法对比又只比整系统，难拆出「哪些选择真影响跟踪」。
- **主张：** 固定 nominal 配置，一次只改一个因素：**(i) motion command**（是否含参考关节速度）**(ii) observation history**（0/10/20）**(iii) action**（残差相对参考 vs 相对默认姿态）**(iv) actuation profile**（力学启发 PD vs 更硬固定尺度）**(v) hand-force randomization** **(vi) Teacher–Student vs 标准 PPO**。
- **产物：** 开源 YAHMP + 仿真消融 + 真机 zero-shot / 抗扰 / 手部持力实验。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-yahmp.md`](../../wiki/entities/paper-yahmp.md)；与 [TWIST2](../../wiki/entities/paper-twist2.md)、[mjlab](../../wiki/entities/mjlab.md)、[Extreme-RGMT](../../wiki/entities/paper-extreme-rgmt.md) 互链。

## 摘录 2：Nominal 与数据（§II）

- **平台：** Unitree G1，29 DoF；控制 50 Hz；仿真 MuJoCo 步长 0.005 s。
- **观测：** 本体 \(p_t\)（基座角速度、重力方向、相对默认姿态关节、关节速度、上一步动作）+ 名义 motion command \(c_t\)（参考关节位姿/速度、平面基座速度、偏航率、高度、roll/pitch）+ 可选历史 \(H=10\)。
- **动作：** 名义为相对参考关节的残差，经 PD 变关节目标。
- **数据：** 过滤后 AMASS+OMOMO 共 **12,175** 条；训练 **11,151** / 测试 **1,024**（分劈在消融前固定）。
- **训练：** 8192 并行环境 × 20k PPO 迭代；单卡 RTX 4090 ≈25 h/run；Teacher–Student 先训特权 teacher，再 KL 正则 student。

**对 wiki 的映射：** 实体页画训练→ONNX→真机时序图；强调「可复现对照框架」定位。

## 摘录 3：仿真与真机要点（§III–§V）

| 发现 | 要点（相对 Nominal；测试集 1024 条全成功） |
|------|---------------------------------------------|
| 参考关节速度 | Pos-ref-only：基座误差 **+7–14%**，关节速度误差 **+15%** → 应显式进 command |
| 观测历史 | No history：基座误差 **+38–50%**；History-20 无一致收益 → **H=10** 足够 |
| 残差动作 | 对 key-body / 关节位姿略好；基座混杂 |
| 驱动剖面 | 更硬固定尺度跟踪增益有限，但全身最大力矩 **+13%**、上肢 **+41%** → 力学启发剖面更省力矩峰值 |
| Teacher–Student | 相对标准 PPO **仅轻微**改善，训练成本更高 |
| vs TWIST2（同数据重训） | TWIST2 key-body 位置略好（−10%）；基座 / 朝向 / 关节误差明显更大（基座 **+32–36%**，关节速度 **+81%**） |
| 真机 | Nominal **zero-shot**（无真机微调）；软垫上仍可蹲/操作式双支撑；手部力随机化使 4 kg 肘偏角 **15.5°→6.6°**，并可撑到 5–6 kg |

**对 wiki 的映射：** 用「设计选择→跟踪 vs 力矩 vs 交互力」表写清选型读法。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-yahmp.md`**（含流程总览 + 源码运行时序图）。
- 新建 **`sources/repos/yahmp.md`**。
- 交叉更新 TWIST2、mjlab、Extreme-RGMT、身体系统栈 overview。
