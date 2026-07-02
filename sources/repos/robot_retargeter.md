# robot_retargeter

> 来源归档

- **标题：** robot_retargeter
- **类型：** repo
- **链接：** https://github.com/ccrpRepo/robot_retargeter
- **中文 README：** https://github.com/ccrpRepo/robot_retargeter/blob/main/README_zh.md
- **入库日期：** 2026-07-02
- **一句话说明：** 开源 SMPL-X / 源机器人 CSV → 多目标人形关节轨迹的重定向工具集：mink + MuJoCo IK、连杆比例缩放、膝部几何重建、接触检测与足端滑动抑制，支持 G1/H2/T800/R1 等并排可视化。
- **沉淀到 wiki：** 是 → [`wiki/entities/robot-retargeter.md`](../../wiki/entities/robot-retargeter.md)

## 摘录要点

### 流水线三步

1. **回放 / 提取关键点**：从 SMPL-X `.npz`（[AMASS](https://amass.is.tue.mpg.de/) 等）或源机器人 LAFAN1 风格 `.csv` 提取骨骼关键点。
2. **重定向**：[mink](https://github.com/kevinzakka/mink) + MuJoCo 多目标 IK，YAML 配置每机型骨架映射。
3. **可视化**：多机器人重定向结果并排播放。

### 输入 / 输出

| 输入 | 说明 |
|------|------|
| SMPL-X `.npz` | 需单独下载 [SMPL-X](https://smpl-x.is.tue.mpg.de/) 模型至 `asset/smplx/`；仓库含 ACCAD 示例子集 |
| 机器人 `.csv` | LAFAN1 格式（根四元数 + 米 + 关节弧度）；含 G1 dance 示例 |
| [Bones Seed G1](https://huggingface.co/datasets/bones-studio/seed) | 根欧拉角/厘米/角度制，需 `convert_bones_to_lafan1.py` 转换后接入 |

| 输出 | 说明 |
|------|------|
| `output_data/robot_motion/` | 目标机器人关节轨迹 CSV |
| 关键点序列 | 缩放 / 接触修正后的中间关键点 |

### 核心机制（README 强调）

- **连杆比例缩放**：按目标机器人连杆长度比逐段拉伸源关键点，**保持各段朝向**；根位移按腿长比缩放步幅。
- **膝部弯曲重建**：两段骨 IK + `knee_angle_offset_degrees`（常用约 15°）提升下肢可达性。
- **接触检测**：手脚低速度 + 低高度双阈值；自适应地面高度 LPF；支撑相足端 **FrameTask 锁定** 抑制 foot sliding。
- **斜胯构型**：t800 等需在髋部额外映射点以稳定骨盆倾斜表达。

### 支持机型（README 示例）

`g1`、`h2`、`t800`、`r1`、`jaka_pi`、`pnd_adam` 等；`VIS_ROBOTS` 环境变量可空格分隔多机并排。

### 一键脚本

- `./bash/retarget_from_smplx.sh` — SMPL-X → 多机器人
- `./bash/retarget_from_robot.sh` — 源机器人 CSV → 目标机器人

### 对 wiki 的映射

- **wiki/entities/robot-retargeter.md**：实体页（mink IK 多机型重定向工具）。
- **wiki/concepts/motion-retargeting.md**：工具与数据集表补充。
- **wiki/concepts/motion-retargeting-pipeline.md**：IK 阶段工程实现参照。
- 与 [SOMA Retargeter](../../wiki/entities/soma-retargeter.md) 共享 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 消费路径，但输入表示与 IK 栈不同（SMPL-X / LAFAN1 CSV vs SOMA BVH + Warp）。
