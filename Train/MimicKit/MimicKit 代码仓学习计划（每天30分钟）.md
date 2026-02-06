这是一个**基于物理的角色动作模仿强化学习框架**，包含 DeepMimic、AMP、ASE、ADD 四种主流算法。以下是为期约 **3 周** 的学习计划：

---

### 第一周：基础概念与整体架构

### Day 1 - 项目概览与运行测试

**目标**：了解项目用途，成功运行一个示例

- 阅读 `README.md` 了解项目整体功能
    
- 安装依赖，下载数据集
    
- 运行一个可视化测试命令验证环境：
    
    ```bash
    python mimickit/run.py --arg_file args/view_motion_humanoid_args.txt --visualize true
    ```
    

### Day 2 - 程序入口与流程

**目标**：理解训练/测试的主流程

- 阅读 `mimickit/run.py`（仅160行）
- 理解关键流程：`load_args → build_env → build_agent → train/test`
- 理解配置文件系统：`engine_config`、`env_config`、`agent_config`

### Day 3 - 配置文件系统

**目标**：理解YAML配置如何驱动整个系统

- 查看 `data/envs/deepmimic_humanoid_env.yaml`
- 查看 `data/agents/deepmimic_humanoid_ppo_agent.yaml`
- 查看 `args/deepmimic_humanoid_ppo_args.txt`
- 理解三层配置的关系

### Day 4 - 基础环境类

**目标**：理解强化学习环境的基本接口

- 阅读 `mimickit/envs/base_env.py`（仅70行）
- 理解核心接口：`reset()`、`step()`、`get_obs_space()`、`get_action_space()`

### Day 5 - 基础智能体类（上）

**目标**：理解RL Agent的基本结构

- 阅读 `mimickit/learning/base_agent.py` 的前半部分（1-200行）
- 重点理解：`__init__`、`train_model`、`_train_iter`

### Day 6 - 基础智能体类（下）

**目标**：理解训练循环的细节

- 阅读 `mimickit/learning/base_agent.py` 的后半部分（200-450行）
- 重点理解：`_rollout_train`、`_decide_action`、`_record_data_*`

### Day 7 - 周末复习与实践

- 复习本周内容，画出整体架构图
- 尝试运行 DeepMimic 训练（可以只跑几分钟看看效果）

---

### 第二周：动作表示与核心算法

### Day 8 - 动作数据格式

**目标**：理解动作数据的表示方式

- 阅读 `mimickit/anim/motion.py`
- 理解：根位置(3D)、根旋转(3D)、关节旋转的表示
- 理解指数映射（exponential map）旋转表示

### Day 9 - 动作库与运动学模型

**目标**：理解动作数据的加载和处理

- 阅读 `mimickit/anim/motion_lib.py`
- 理解 `calc_motion_frame` 如何插值获取任意时刻的姿态
- 阅读 `mimickit/anim/kin_char_model.py` 了解正向运动学

### Day 10 - 角色环境基类

**目标**：理解物理角色的状态表示

- 阅读 `mimickit/envs/char_env.py`（重点看 `_compute_obs` 和状态获取）
- 理解：root_pos, root_rot, dof_pos, dof_vel, body_pos 等概念

### Day 11 - DeepMimic 环境（上）

**目标**：理解 DeepMimic 的核心思想

- 阅读 DeepMimic 论文摘要和 `docs/README_DeepMimic.md`
- 阅读 `mimickit/envs/deepmimic_env.py` 的前半部分
- 理解参考动作跟踪机制

### Day 12 - DeepMimic 环境（下）

**目标**：理解奖励函数设计

- 阅读 `deepmimic_env.py` 中的 `compute_reward` 函数
- 理解各个奖励项：pose_r, vel_r, root_pose_r, root_vel_r, key_pos_r
- 理解 `compute_done` 的终止条件

### Day 13 - PPO 智能体

**目标**：理解 PPO 算法实现

- 阅读 `mimickit/learning/ppo_agent.py`
- 重点理解：`_decide_action`、`_compute_actor_loss`、`_compute_critic_loss`
- 理解 PPO clip 机制

### Day 14 - 周末复习与实践

- 复习 DeepMimic 的完整流程
- 尝试修改奖励权重，观察训练效果变化

---

### 第三周：高级算法与扩展

### Day 15 - AMP 环境

**目标**：理解对抗式动作先验

- 阅读 AMP 论文摘要和 `docs/README_AMP.md`
- 阅读 `mimickit/envs/amp_env.py`
- 理解 discriminator 观测的构建方式

### Day 16 - AMP 智能体与判别器

**目标**：理解 AMP 的对抗训练机制

- 阅读 `mimickit/learning/amp_agent.py`
- 重点理解：`_compute_disc_loss`、`_compute_disc_reward`
- 理解如何用判别器奖励替代手工设计的奖励

### Day 17 - ASE 环境与智能体

**目标**：理解技能嵌入方法

- 阅读 ASE 论文摘要和 `docs/README_ASE.md`
- 阅读 `mimickit/envs/ase_env.py`
- 阅读 `mimickit/learning/ase_agent.py`
- 理解 latent skill embedding 的概念

### Day 18 - ADD 方法

**目标**：理解可微判别器

- 阅读 ADD 论文摘要和 `docs/README_ADD.md`
- 阅读 `mimickit/envs/add_env.py`
- 阅读 `mimickit/learning/add_agent.py`
- 理解与 AMP 的区别

### Day 19 - 神经网络架构

**目标**：理解模型结构

- 阅读 `mimickit/learning/ppo_model.py`
- 阅读 `mimickit/learning/amp_model.py`
- 浏览 `mimickit/learning/nets/` 下的网络定义

### Day 20 - 物理引擎接口

**目标**：理解多引擎支持

- 阅读 `mimickit/engines/engine.py`
- 了解 Isaac Gym / Isaac Lab / Newton 的接口抽象
- 理解 Engine 如何封装物理仿真

### Day 21 - 总结与进阶

**目标**：整体回顾和进阶方向

- 回顾整体架构，整理笔记
- 了解工具脚本：`tools/gmr_to_mimickit/`、`tools/smpl_to_mimickit/`
- 规划后续：自定义动作、自定义角色、自定义任务

---

### 学习建议

1. **边读边注释**：在代码中添加中文注释帮助理解
2. **画流程图**：每完成一个模块，画出数据流图
3. **动手实验**：修改参数观察效果变化是最好的学习方式
4. **论文配合**：建议同步阅读对应论文，尤其是方法部分

### 关键论文（按顺序阅读）

1. **DeepMimic (2018)** - 基础方法
2. **AMP (2021)** - 对抗式方法，无需相位变量
3. **ASE (2022)** - 技能嵌入，可复用
4. **ADD (2025)** - 最新改进