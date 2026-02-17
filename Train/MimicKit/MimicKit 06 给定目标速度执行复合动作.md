现在我对所有配置有了全面的了解。以下是分析结果：

---

在本框架的 `args/` 目录中，**给定目标速度执行复合动作**的训练任务配置有以下 **4 个**：

## 1. `amp_steering_humanoid_args.txt` — Humanoid 转向控制

- **环境**: `task_steering`
- **角色**: humanoid
- **目标速度**: 随机范围 `tar_speed_min: 0.5` ~ `tar_speed_max: 5.0`
- **目标方向**: 随机目标方向 (`rand_tar_dir: True`)，不随机面朝方向
- **动作集**: `dataset_humanoid_locomotion.yaml`（运动类动作集合）
- **奖励**: 包含 steering 速度跟踪奖励 (`reward_steering_tar_w`, `reward_steering_vel_scale`)

## 2. `amp_steering_humanoid_sword_shield_args.txt` — Sword & Shield 角色转向控制

- **环境**: `task_steering`
- **角色**: humanoid_sword_shield
- **目标速度**: 随机范围 `tar_speed_min: 1.4` ~ `tar_speed_max: 1.7`
- **目标方向**: 随机目标方向 + 随机面朝方向 (`rand_tar_dir: True`, `rand_face_dir: True`)
- **动作集**: `dataset_humanoid_sword_shield_locomotion.yaml`
- **奖励**: 包含 steering 速度跟踪奖励

## 3. `amp_location_humanoid_args.txt` — Humanoid 目标位置导航

- **环境**: `task_location`
- **角色**: humanoid
- **目标速度**: 固定 `tar_speed: 1.0`
- **目标位置**: 最远距离 `tar_dist_max: 10.0`，随机切换间隔 5~10s
- **动作集**: `dataset_humanoid_locomotion.yaml`

## 4. `amp_location_humanoid_sword_shield_args.txt` — Sword & Shield 角色目标位置导航

- **环境**: `task_location`
- **角色**: humanoid_sword_shield
- **目标速度**: 固定 `tar_speed: 1.0`
- **目标位置**: 最远距离 `tar_dist_max: 10.0`，随机切换间隔 5~10s
- **动作集**: `dataset_humanoid_sword_shield_locomotion.yaml`

---

**总结**：这 4 个任务都属于 AMP（Adversarial Motion Priors）框架下的**任务导向型训练**，它们共同的特点是：
- 给定**目标速度**（steering 任务为速度范围，location 任务为固定速度）
- 从**运动数据集**（locomotion dataset）中学习多种运动风格的**复合动作**
- 使用 `amp_task_humanoid_agent` 作为 agent，结合 AMP 判别器奖励和任务奖励进行训练

其余配置（如 `deepmimic_*`、`amp_*` 无 steering/location 前缀、`ase_*`、`vault_*`）要么是单纯的动作模仿，要么是环境交互（如翻越障碍物），并不涉及给定目标速度的复合动作控制。