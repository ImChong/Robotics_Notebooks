# HumanX（wyhuai/humanx）官方 XGen / XMimic 源码

> 来源归档（repo / 官方实现）

- **仓库：** <https://github.com/wyhuai/humanx>
- **论文：** <https://arxiv.org/abs/2602.02473>（CoRL 2026）
- **项目：** <https://wyhuai.github.io/human-x/>
- **核对日期：** 2026-11-18
- **核对内容：** 根 README、xgen/README.md、xmimic/README.md、xgen/xgen_editor/run_hoi_editor.sh、xgen/xgen_editor/hoi_editor/server.py、xmimic/tools/convert_xgen_hoi_npz.py、xmimic/humanoidverse/train_agent.py、eval_agent.py。
- **状态：** XGen 编辑器、数据转换器与 XMimic 仿真训练/评测实现已开源；**不含完整真机部署链**。
- **许可：** README 标示 Apache-2.0 WITH Commons-Clause，需留意附加限制。

## 官方数据链与可辨识入口

1. **视频到机器人参考动作**：官方推荐外部 GVHMR 恢复单目人体动作、UMR 重定向到 G1；XGen 接收已准备好的机器人动作数据，并不自带这两个外部模块。
2. **启动 XGen**：在 xgen/ 下安装 `numpy`、`mujoco>=3.15`、`joblib`，运行 `bash xgen_editor/run_hoi_editor.sh --port 8766`；shell `exec python xgen_editor/hoi_editor/server.py`，本地浏览器 WebGL 编辑并导出独立 HOI `.npz`。
3. **运动格式转换**：在 xmimic/ 下运行 `python tools/convert_xgen_hoi_npz.py --input-dir data/motions/speed_capped_5mps --output data/motions/xgen_speed_capped_5mps/speed_capped_5mps_dynamic.pkl --no-freeze-until-release`。源码处理 G1 29DoF 关节顺序、四元数、物体运动/接触、skill label。
4. **教师训练**：`python humanoidverse/train_agent.py +simulator=isaacgym_hoi +exp=humanx/hoi_teacher +domain_rand=domain_rand_low +rewards=humanx/hoi +robot=g1/g1_29dof_model16 +terrain=terrain_locomotion_plane +obs=humanx/hoi_teacher_tracking num_envs=4096 robot.motion.motion_file=data/motions/example/basketball_catch_throw.pkl`；完整 Hydra 覆盖及配置参见 XMimic README。源码调用 `instantiate(config.env)`、`instantiate(config.algo)`、`algo.setup()` 与 `algo.learn()`。
5. **学生蒸馏**：同一个 train_agent.py 使用 `+exp=humanx/hoi_DAggerBC`、`+obs=humanx/mocap/hoi_DAggerBC` 和 `algo.config.teacher_ckpt_path=...`；教师也有 AMP-PPO 配置。
6. **评测**：`python humanoidverse/eval_agent.py checkpoint=logs/HumanX/<run_dir>/model_<iter>.pt headless=false num_envs=1 export.policy=false export.onnx=false`；源码加载 checkpoint、`algo.evaluate_policy()`，可选 TorchScript/ONNX。

## 环境、可复现范围与边界

- XGen Python 3.10+、MuJoCo 与 WebGL；XMimic Python 3.8、NVIDIA Isaac Gym Preview 4、兼容 CUDA/PyTorch，依赖需另装。
- 官方 README 声明目前仿真示例涉及篮球传接、假动作投篮、足球踢球、箱/筐拾放；并支持用自有视频流程训练新的交互技能。
- XMimic README 明确未包含完整硬件部署、ROS/Unitree 控制或 sim-to-real calibration，不能将论文真机演示视为仓库提供了对应完整部署代码。
- **未执行** GPU 训练、实际仿真评测、复现指标或真机部署；所述入口是官方文档与源码静态核验。

## 对 wiki 的映射

- [HumanX 独立论文详情](../../wiki/entities/paper-hrl-stack-05-humanx.md)
- [HumanX 官方项目页](../sites/humanx-corl2026-project.md)
