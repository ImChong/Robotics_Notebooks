# Flash ALOHA — 官方源码（JohnYao93/flash-aloha）

> 来源归档（源码与运行入口，2026-11-13 核查）

- **标题：** Flash ALOHA
- **类型：** repo
- **官方仓库：** <https://github.com/JohnYao93/flash-aloha>
- **论文：** <https://johnyao93.github.io/static/files/flash-aloha.pdf?v=2>（[论文归档](../papers/flash_aloha_icra2026.md)）
- **项目页：** <https://johnyao93.github.io/flash-aloha/>（[站点归档](../sites/flash-aloha.md)）
- **作者 / 维护者：** John Hua Yao 等；JohnYao93
- **许可证：** MIT（仓库 LICENSE）
- **开源情况：** 已开源：采集、训练、推理；论文原始实验示范数据和完整 checkpoints 未在 README 发现一键下载入口
- **入库日期：** 2026-11-13
- **一句话说明：** 可反驱 ARX 机械臂动觉示范采集和录制回放，ACT/Flow 训练，动作块融合与真机推理的公开工程实现。
- **源码核对范围：** README、`scripts/run.py`、`src/config/run.py::build_plan`、`scripts/record.py`、`src/recording/record_episode_single.py` 的录制/状态读取与回放路径、`scripts/policy/imitate_episodes_train.py`、`src/policy/imitate_episodes_train.py::main_train`、`scripts/infer.py`、`src/inference/inference_direct.py`（`blend_actions`、硬件动作发送）、`src/inference/policy_chunks.py` 与 `src/contracts/standard_image_processing.py`。
- **通读：** 已完成

## 代码入口（按实际 run.py 路由）

| 用户入口 | 解析后的脚本 | 实际工作 |
|---|---|---|
| `python scripts/run.py collect single_arm` | `scripts/record.py single` → `src/recording/record_episode_single.py` | 在被动/引导模式采机械臂关节状态与相机帧，存储 episode，支持回放核验 |
| `python scripts/run.py train single_arm_act` | `scripts/policy/imitate_episodes_train.py` → `src/policy/imitate_episodes_train.py` | 根据 YAML 配置读取数据、构建 ACT/Flow 策略，保存检查点 |
| `python scripts/run.py infer single_arm_act` | `scripts/infer.py direct` → `src/inference/inference_direct.py` | 读相机/关节状态，策略产生动作块，执行段边界融合与关节目标下发 |
| `python scripts/run.py serve single_arm_act` | `scripts/infer.py serve` → `src/inference/serve_policy.py` | 可选服务端；论文主要结果不以此部署模式为实验前提 |

`scripts/run.py` 为执行 `src/config/run.py` 的轻量包装；后者 `build_plan` 负责按工作流和训练/任务配置生成真实命令。入口名字来自已检查的 `src/config/run.py`，执行需有匹配的配置和硬件；不要把上表误当作无需配置的一键复现命令。

## 对源码流程的核对

1. **采集：** `scripts/record.py` 的 `single` 模式路由到 `src/recording/record_episode_single.py`；该文件通过 `read_robot_state`、`standard_capture.capture_observation` 取得关节位置/速度与相机帧，形成 episode 并写盘。操作员握臂的物理示教系统还依赖 ARX SDK 与真机配置。
2. **训练：** `src/policy/imitate_episodes_train.py` 核对任务 YAML、录制文件、动作数据契约，选 `ACT` 或 `Flow` 构造策略，执行训练及 checkpoint 管理。论文定量实验对应 **ACT**，不能用仓库里同时存在的 Flow 证明论文做过 Flow 对照。
3. **执行：** `src/inference/inference_direct.py` 导入 `send_joint_target` 控制接口，`blend_actions` 支持 `linear`、`exponential` 与 `sigmoid`，其中 sigmoid 对重叠块同一执行时刻的旧预测和新预测加权融合。
4. **图像：** `src/contracts/standard_image_processing.py` 维护训练/推理一致的图像规范。**本次核对尚未定位论文所述按 FK+标定投影操作员手部并涂黑的完整源码函数**；不能把一般图像尺寸/色彩预处理直接称为论文的几何遮罩实现。
5. **运行边界：** 代码中的 ARX 工作流已核对；论文抛接球实验使用 HESSIAN 机械臂，不能推定当前代码仓的 ARX 入口可直接复现该硬件实验。实际运行还需硬件 SDK、相机和配置、训练数据或模型。

## 与 wiki 的关系

- [唯一论文详情页](../../wiki/entities/paper-flash-aloha.md)
- [双臂操作方法入口](../../wiki/tasks/bimanual-manipulation.md)
