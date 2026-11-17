# FORGE：力感知多粒度人形移动操作（arXiv:2611.17142）

> 一手论文核对归档；对应的 wiki 独立论文节点已在仓库内存在，本次原位补全

- **英文标题：** FORGE: Force-Aware Multi-Granularity Control for Efficient Humanoid Loco-Manipulation
- **作者：** Liangjun Xing, Ke Fan, Kailin Li, Bin Wang, Xiaojie Niu, Qingwei Ben, Jiafei Cao, Wei Yao, Yebin Liu, Jingbo Wang, Weinan Zhang
- **机构：** 上海人工智能实验室、清华大学、浙江大学、上海交通大学、香港中文大学
- **arXiv：** https://arxiv.org/abs/2611.17142（v1，2026-11-12）
- **HTML 全文：** https://arxiv.org/html/2611.17142v1
- **PDF：** https://arxiv.org/pdf/2611.17142v1
- **官方项目：** https://bb0928.github.io/forge/
- **官方代码：** https://github.com/bb0928/FORGE
- **入库日期：** 2026-11-16；本次复核：2026-11-17
- **核对范围：** arXiv HTML v1 全文 I–V：方法 §III-A/B/C、公式 (1)–(6)、网络结构、训练设置、表 I–V、全部评测设置及实验边界；官方项目页全部 Research 小节，以及 README、实际训练和仿真评测入口源码。
- **通读：** 已完成
- **一句话说明：** 在冻结行走先验上训练全身残差策略，以导航/站姿/腕位姿/期望腕力四级命令的激活掩码切换任务，并用动量观测器加学习补偿估计低频腕部接触力，进而调整虚拟腕位姿目标。

## 原文事实与方法

1. **四组命令与掩码（§III-A）：** base velocity / planar feet positions / bilateral wrist poses / optional robot-applied wrist-force targets；未激活组的观测清空、奖励关闭。六帧观测历史送入 MLP + 两层 Transformer 和 adaptation encoder；动作维度 29，策略频率 50 Hz。训练不需要整个人体动作或关节角参考轨迹。
2. **传感器替代路径（§III-B）：** 由双臂关节状态与重构的低层 PD 力矩代理更新 14 维动量观测残差，经腕部平移雅可比投影成 3D 等效力；再用仿真接触力监督的 6D 学习残差补偿。仿真真实力只用于训练，真机没有腕部六维力/力矩传感输入。
3. **虚拟目标控制（公式 5）：** 以**机器人施加力** `f_cmd` 为期望量、**环境反作用力** `f_env` 为估计量，更新 `p_virtual = p_nominal + K_f^-1 (f_env_hat + f_cmd)`。是准静态的类导纳位姿目标调整，不是动力学全阻抗或直接力矩控制；`K_f=200 N/m` 为文中力命令模式示例。
4. **不对称残差（§III-C）：** 冻结仅下身的 locomotion expert，下肢输出 `a_base_lower + delta_a_lower`，上肢为 `delta_a_upper`；PPO 对下肢探索标准差限制为 0.2（上肢 1.0）来保护行走先验。
5. **工程条件（§IV-A）：** 论文写 Isaac Gym 200 Hz、每四个仿真步执行一次 50 Hz 策略、2,048 环境、最多 20,000 轮/阶段；公开第二阶段 shell 默认 4,096 环境，二者不是同一套原样参数。

## 原始评测及不可混淆的口径

| 来源 | 实验类型 | FORGE 结果及解释 |
|---|---|---|
| Table I | **真机**四控制器各 20 次 | 命令发出到末端到达目标的延迟 **22 ms**；足 MPE **5.42 cm**；腕 MPE **4.15 cm**。22 ms **不是**神经网络纯推理耗时 |
| Table I | **仿真**同腕位姿目标双臂挤压刚性箱体 | 平均法向反作用力 **8.52 N**，较 HOMIE 15.48、TWIST2 17.16、SONIC 15.32 N 小；衡量该特定条件下挤压力，不是力跟踪误差 |
| Table III | 仿真激活掩码消融，20 次 | 有效 rollout **19.46 ± 0.52 s**，去掩码 **15.69 ± 1.32 s**；腕 MPE 分别 **0.079 ± 0.004 m**、**0.115 ± 0.003 m** |
| Table IV / Fig. 8 | 真机阶梯外加载荷的**代表性曲线** | Hybrid x/y/z RMSE 分别 **0.81/1.97/0.87 N**；不是全部十次重复的汇总误差。35/40/50 N 未见载荷的 RMSE 达 **8.28/9.66/12.35 N** |
| Table V | 真机 Diffusion Policy → FORGE 搬箱（每载荷 20 次） | 训练共 **320 条**跨载荷示范；0、0.5、1、2 kg 的成功率 **100%、100%、90%、90%** |

## 对比与结论边界

- 控制方式在 FORGE 中是一种统一的任务空间执行接口；上游 Diffusion Policy 只是应用示例，不等于 FORGE 本身是 VLA/WAM。
- Fig. 9 的泡沫箱实验展示开启力反馈时可见变形减轻；不是通用多物体力控成功率证明。
- 低频腕部 3D 等效力估计不等于 6D wrench 传感、瞬态冲击估计或全空间精确力控。
- 已开放第二阶段训练、Isaac Gym 回放；第一阶段基础策略训练、MuJoCo 及真机部署尚未开放，无法只凭目前源码独立完整复现真机实验。
- 本次为一手论文、项目页、源码的**静态核对**，未实际训练/运行仿真/部署真机。

## 来源交叉归档

- [官方项目页核对](../sites/forge.md)
- [官方源码核对](../repos/forge.md)

## 对 wiki 的映射

- [FORGE 论文唯一实体节点](../../wiki/entities/paper-forge-force-aware-loco-manipulation.md)
