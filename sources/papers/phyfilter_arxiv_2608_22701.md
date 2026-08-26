# PhyFilter（物理滤波残差修正以换泛化）

> 来源归档（ingest）

- **标题：** Physics Filtering Favors the Generalization of Robot Learning
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.22701>
  - <https://scoardyy.github.io/PhyFilter>
- **代码：** <https://github.com/JIAjindou/PhyFilter>
- **机构：** 南洋理工大学（NTU）；北京航空航天大学（BUAA）；新加坡国立大学（NUS）
- **入库日期：** 2026-08-26
- **一句话说明：** 用可插拔低通物理滤波去捕捉学习残差的低频分量，结合实时状态反馈与运动学/动力学结构修正 RL 或监督学习输出；参数可用伴随梯度自动搜索，不必为人调极点。

## 核心摘录（MVP）

### 1) 机器人不能靠堆数据复制 LLM 规模

- **摘录要点：** 真机示教吞吐有限，仿真/视频合成仍有 sim2real 与格式分裂。生物泛化依赖身体结构与反馈，而非单纯缩放。PhyFilter 把学习残差 \(\gamma=f-f_\theta\) 经滤波器 \(\mathcal{F}\) 加回：\(\hat f=f_\theta+\mathcal{F}(\gamma)\)。
- **对 wiki 的映射：**
  - [PhyFilter](../../wiki/entities/paper-phyfilter.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 即插即用 vs 重训 PINN

- **摘录要点：** 多数物理信息网络要改结构并重训；PhyFilter 是轻量模块，可叠在预训练策略/估计器上。四案例：四足 RL（Isaac Gym 关节层）、无人机 SEER-I 动力学、空中操作耦合动力学、加速度微分感知。
- **对 wiki 的映射：**
  - [PhyFilter](../../wiki/entities/paper-phyfilter.md) — 方法。
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 3) 四系统数字

- **摘录要点：** 四足仅在仿真平地训练即可过真机石板/草坪/沙地/碎石；基线在沙/砾立即摔倒。无人机圆轨迹 + 质量与风扰：相对 SEER-I MAE **↓30.22%**、相对基线 **↓50.17%**。空中臂 5 m/s 风 + 0.3 kg 质量不确定下端执行误差上限 2.5 cm（相对基线均值/方差改善约 24%/55%）。部署关掉修正仍能过部分真机地形，说明训练期滤波也塑造了更好策略。
- **对 wiki 的映射：**
  - [PhyFilter](../../wiki/entities/paper-phyfilter.md) — 评测。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** **已开源**。`JIAjindou/PhyFilter` 含 `quadruped_case`（`train.py` / `play.py`）、`drone_case` / `aerial_manipulation_case` / `acceleration_case`（MATLAB/Simulink）与 `auto_learning`。四足 PhyFilter 训练需在 `legged_robot.py` 取消注释指定行。
- **对 wiki 的映射：**
  - [仓库归档](../repos/phyfilter.md)
  - [项目页](../sites/phyfilter-scoardyy.md)

## 当前提炼状态

- [x] arXiv HTML + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-phyfilter.md` 新建
