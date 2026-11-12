# FRDV：Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning

> 来源归档（paper）：马萨诸塞大学阿默斯特分校 DARoS Lab，Humanoids 2026

- **标题：** Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning
- **中文标题：** 面向可扩展人形机器人学习的快速动作重定向流水线
- **作者：** Daniel Marew、Sangjun Park、Nisal Perera、Donghyun Kim
- **论文：** <https://arxiv.org/abs/2610.28470>
- **PDF：** <https://arxiv.org/pdf/2610.28470>
- **版本：** arXiv:2610.28470v1，2026-10-20；Humanoids 2026
- **项目页：** <https://daroslab.github.io/frdv/>
- **代码：** <https://github.com/DARoSLab/FRDV>
- **入库日期：** 2026-10-26；本次全文复核：2026-11-12
- **核对范围：** arXiv v1 PDF 全文 8 页：引言、相关工作、`III 方法与算法 1–4、`IV 实验与结果表、`V 结论及参考文献；另与官方项目页、2.0 README 和源码入口交叉核验。
- **通读：** 已完成

## 一句话摘录

FRDV 采用可并行的约束 IK 快速生成大规模人形参考轨迹；对需要进一步验证的动作，在独立的 CI-iLQR 阶段用完整刚体动力学检查跟踪可行性，并修正部分接触伪影。两阶段计算开销不能混为一谈。

## 主要方法（以论文 v1 为准）

1. **`III-A，几何目标与约束：** MoCap 关键点依据机器人身体比例转换为 site 目标；目标函数结合关键点位置误差、姿态正则及时间平滑；关节限位、脚/地面、胶囊碰地和自碰撞作为约束。
2. **`III-B，快速优化：** 自定义 Gauss–Newton SQP 线性化约束，QP 在论文 1.0 中使用 OSQP，结合 L1 merit 回溯搜索；RTI 热启动后续帧以减小优化开销。
3. **`III-C，批处理：** 动作分块并行，从每块冷启动解算；拼接边界在启用 refine 时按照速度/加速度异常检测再局部优化；性能须区分是否 refine。
4. **`III-D，接触隐式验证：** 将参考动作交给完整 MuJoCo 动力学系统，用 CI-iLQR 测试跟踪，纳入驱动与接触影响；部分脚悬空、滑动被修复，也可能发现无法执行的参考。
5. **`IV，系统验证：** 比较 GMR、ProtoMotions 等运动学基线，训练运动跟踪 RL 策略评估可跟踪性；展示 Unitree G1 的 Sim2Real。

## 实验指标及限定条件

- **1.0 运动学吞吐：** LAFAN1 使用 20 workers、关闭边界 refine 时报告最高 **9,156 FPS**；**不含 CI-iLQR 阶段**。
- **跨动作库处理：** 论文使用 AMASS 子集大规模数据；不同系统对动作的裁剪及处理帧数可能不同，不能只看整库小时数跨方法算“等任务加速”。
- **动力学精炼个案：** Dance1-S1 选定 26 秒中，全局身体位置误差 **103.12→86.58 mm**，anchor-relative 误差 **29.26→30.57 mm**；不能宣称所有质量指标均提升。
- **限制：** 论文动力学验证不是每条动作都无代价运行，也不保证修正所有不合理动作。单次轨迹经过 IK 仍可能失败于动力学检查。

## 与 FRDV 2.0 开源版本的关系

- **项目是同一个，软件版本不同。** 官方仓库 `pyproject.toml` 当前为 `2.0.0`。2.0 默认多文件 `mode=files` 并行，原生 C / Numba 整段计算；`mode=chunks` 要求显式选择。
- **QP 后端不同。** 2.0 `src/frdv/qp.py` 的实现是 Goldfarb–Idnani 双主动集 QP，不能将论文中 OSQP 的描述原封不动应用于 2.0。
- **不能归因于动力学验证的性能提升：** README 和 2026-11-08 基准报告的 **503,288.6 FPS** 是 LAFAN1→G1 运动学流水线、Apple M1 8 workers 的热运行处理吞吐，排除保存、启动及编译；动力学模块没有公开。
- **开源边界：** 官方 README 明确：`The dynamics-validation pipeline described in the paper is not included in this repository yet.`

## 交叉来源与知识映射

- [官方项目页归档](../sites/frdv-project.md)
- [2.0 官方代码与测量口径](../repos/frdv.md)
- [wiki 唯一 FRDV 实体节点](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md)
- [原始论文 PDF](https://arxiv.org/pdf/2610.28470)
