# LpWM：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2608.22764>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/YilunKuang/lpworldmodel>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2608.22764 HTML（2026-11-15 获取）正文 §1–5、附录 A–H（相关工作、预测器、环境、两套评测设置、Jaccard、muP、线性化证明及超参数扫描）；阅读全部正文实验表与附录表/图注。另核对官方 README、train.py 入口及 visual_world_model.py 的 _forward_adaln/_rollout_adaln；未训练模型。
- **通读：** 已完成

本次复核单独存档；[历史摘录](lpwm_arxiv_2608_22764.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- [原文](https://arxiv.org/html/2608.22764)的 one-hot 结果依赖紧致状态空间与 Lipschitz 动力学，误差随维度按 O(N^(-1/d)) 收敛，仍有维度灾难；分布式稀疏是实用放松，并不继承全局严格线性保证。LTI 实例的 RepReLU 也使其不是纯线性系统。
- RDMReg 对随机投影的一维分布用2-Wasserstein匹配整流广义高斯；默认p=1、mu=0、sigma=sqrt(1/2)。RepReLU前向ReLU、反向GeLU只是避免死神经元的优化保护，防坍塌仍靠分布正则。
- 附录D区分两套管线：§3 Wall/PushT宽度与预测器扫描，§4 Piecewise/Cube固定小模型；架构、优化器与依赖环境不同，不能直接混合消融。
- CEM每轮300候选、30轮、保留30，H=5、frameskip=5；9000是候选序列评分总数，目标距起点25原始步。开环与每5步重规划的闭环分开报告。
- 附录H训练只有一个seed、规划评测三seed；稠密LeWM控制组也改用2-Wasserstein分布匹配而非原版SIGReg的Epps–Pulley损失。57个百分点不能写成原论文LeWM固定配方下的普遍增益。
- 无TJ时support与末端运动相关0.87、与接触仅0.05；可选TJ改变可解释因子，不等于稀疏性自动发现接触。
- [官方源码](https://github.com/YilunKuang/lpworldmodel)的 `train.py:main` 调用Trainer；`models/visual_world_model.py:_forward_adaln` 对编码与预测同时应用link，再算latent损失和regularizer，`_rollout_adaln`复用link递推。README将 `lpwm_swm/` 明确列为独立环境。

## 对 wiki 的映射

- [LpWM 详情](../../wiki/entities/paper-lpwm.md)
