# motion_control_projects

> 来源归档（ingest）

- **标题：** 开源运动控制项目（飞书公开文档）附带 PDF 集合
- **类型：** paper
- **来源：** 飞书公开文档 + 文中列出的 PDF 附件
- **入库日期：** 2026-04-18
- **最后更新：** 2026-04-18
- **一句话说明：** 把一份面向人形/足式机器人运动控制的公开项目清单整理为可追踪的资料层输入，覆盖训练机制优化、模仿学习、世界模型、物体交互、跑酷与动作重定向。

## 来源上下文

- **原始入口：** 飞书公开文档《【开源】小而美的运动控制项目》
- **文档定位：** 持续更新的运动控制项目笔记，按“训练机制优化 / Parkour / 动作模仿 / 物体交互 / 重定向”组织材料
- **说明：** 当前以“文中可见 PDF 文件名 + 文档正文可见摘要”为准入库；未直接展开的 PDF 先记录为待精读来源

## 核心资料摘录

### 1) HALO.pdf
- **所属主题：** 强化学习训练机制优化 / base 位姿误差修正
- **文中提炼：** 浮动基人形机器人仅靠关节编码器无法稳定恢复 base 在世界坐标系中的位姿；通过每个时间步求解约束 QP，同时修正根部位姿和下肢关节角，使固定脚约束与非穿地约束同时成立，可得到运动学一致、可用于后续系统辨识的全局轨迹。
- **对 wiki 的映射：**
  - [Floating Base Dynamics](../../wiki/concepts/floating-base-dynamics.md)
  - [State Estimation](../../wiki/concepts/state-estimation.md)
  - [System Identification](../../wiki/concepts/system-identification.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 2) 稳健且广义的人形运动跟踪.pdf
- **所属主题：** Actor 网络优化 / Transformer 历史编码
- **文中提炼：** 用因果 Transformer 编码最近 K+1 步本体感知观测，再通过交叉注意力从过去-现在-未来的参考指令窗口中选择当前最相关的子目标；核心价值是把“当前物理状态”和“参考动作片段”对齐，而不是只喂单帧观测。
- **对 wiki 的映射：**
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 3) OmniTrack.pdf
- **所属主题：** 奖励函数设计 / 平滑性正则
- **文中提炼：** 根据参考关节速度对动作平滑惩罚项进行自适应缩放：快动作阶段允许更大的控制变化，慢动作或准静态阶段加强平滑约束，在追踪精度与物理稳定性之间动态折中。
- **对 wiki 的映射：**
  - [Reward Design](../../wiki/concepts/reward-design.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 4) HumanX.pdf
- **所属主题：** 模仿奖励 / 接触图奖励 / 教师学生蒸馏
- **文中提炼：**
  - 用 AMP 判别器奖励提升动作自然度与平滑性
  - 用接触图模仿奖励约束机器人与物体的接触模式
  - 用多个特权教师蒸馏学生策略，证明仅靠本体感知历史也能学习到对外力的估计与应对
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Reward Design](../../wiki/concepts/reward-design.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Contact Estimation](../../wiki/concepts/contact-estimation.md)

### 5) 逐际动力PVP.pdf
- **所属主题：** Parkour / 高动态 locomotion
- **当前状态：** 文档中列为 Parkour 类代表材料，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 6) PHP.pdf
- **所属主题：** Parkour / 高动态 locomotion
- **当前状态：** 文档中列为 Parkour 类代表材料，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 7) Deep Whole-body Parkour.pdf
- **所属主题：** Parkour / 全身动态运动
- **当前状态：** 文档中列为 Parkour 类代表材料，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 8) hiking-in-the-wild.pdf
- **所属主题：** 野外复杂地形 locomotion
- **当前状态：** 文档中列为 Parkour / 复杂地形代表材料，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 9) Beyondmimic.pdf
- **所属主题：** 通用模仿学习基座
- **文中提炼：**
  - 统一任务空间跟踪奖励：位置 / 朝向 / 线速度 / 角速度
  - 通过更精确的 armature 与 PD 增益建模缩小 sim2real gap，替代大量粗糙域随机化
  - 用失败率驱动的自适应采样，把训练 reset 的起点偏向高失败片段，提高数据利用效率
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 10) OmniRetarget.pdf
- **所属主题：** 动作模仿 / 重定向
- **当前状态：** 文档中列为动作模仿类来源，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 11) AMS.pdf
- **所属主题：** 混合奖励 + 动作过滤
- **文中提炼：**
  - 普通动作使用通用跟踪奖励
  - 极端平衡动作才激活质心投影、脚底接触一致性、脚滑惩罚等平衡先验奖励
  - 先做运动学修正，再做平衡约束筛选，构造物理可行的合成平衡运动数据集
- **对 wiki 的映射：**
  - [Reward Design](../../wiki/concepts/reward-design.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Contact Dynamics](../../wiki/concepts/contact-dynamics.md)

### 12) HAIC.pdf
- **所属主题：** 物体交互 / 世界模型教师学生训练
- **文中提炼：** 教师策略、世界模型、学生策略端到端联合训练；第二阶段冻结教师 actor、继续更新世界模型与学生 actor，使学生在仅使用可部署输入的情况下复用特权训练信号。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 13) NVIDIA_VIRAL.pdf
- **所属主题：** 物体交互 / 特权训练与部署
- **当前状态：** 文档中列为物体交互类来源，但当前可见正文未展开具体方法，先保留为待精读来源。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 14) GMR.pdf
- **所属主题：** 通用动作重定向
- **文中提炼：** GMR 主要解决几何层的运动学匹配问题，擅长复现姿态，但缺少动力学、接触力、惯性与可行力矩约束，因此容易出现脚悬空、速度跳变、自碰撞等问题；这明确提示“retarget 后还需要动力学一致化层”。
- **对 wiki 的映射：**
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
  - [Floating Base Dynamics](../../wiki/concepts/floating-base-dynamics.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

## 从文档中额外提炼出的非 PDF 方法线索

这些内容在当前飞书页正文可见，但未明确对应单独 PDF，先作为后续扩展方向：
- **RGMT / Any2Track：** 历史编码器 + 世界模型 + 适配器，强调时序扰动建模与参考动作选择
- **OmniXtreme：** 基于负功率安全的执行器正则化，把硬件安全约束直接写进奖励
- **接触图奖励 / 外力估计：** 对 manipulation 来说，接触模式本身就是应该被监督的目标

## 当前提炼状态

- [x] 记录飞书文档中可见的 14 个 PDF 文件名
- [x] 为已展开的方法（HALO / RGMT / OmniTrack / HumanX / Beyondmimic / AMS / HAIC / GMR）补充摘要
- [~] Parkour 与部分 manipulation 来源（PVP / PHP / Deep Whole-body Parkour / hiking-in-the-wild / OmniRetarget / NVIDIA_VIRAL）仍待单独精读
- [~] 若后续能拿到 PDF 原文或更完整页面，再补 DOI / arXiv / 项目主页链接