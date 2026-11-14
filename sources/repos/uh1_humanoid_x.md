# sihengz02/UH-1 官方仓库源码核对归档

- **仓库地址：** <https://github.com/sihengz02/UH-1>
- **配套资产：** <https://huggingface.co/datasets/USC-GVL/Humanoid-X> · <https://huggingface.co/USC-GVL/UH-1>
- **许可证情况：** 仓库根目录**未提供独立 `LICENSE` 文件**；内嵌子目录 `legged_gym/LICENSE` 与 `rsl_rl/LICENSE` 为 NVIDIA / ETH Zurich 的 `BSD-3-Clause` 开源协议，原论文 Appendix A 声明互联网视频相关数据限学术研究非商用。
- **开源完备度：** **核心训练/推理/重定向代码已开源，上游视频挖掘与标注流水线未开源（标注为 Work in progress）**。
- **核对范围：** 通读官方仓库 `UH-1` 下 `README.md`、`README-Humanoid-X.md`、`option.py`、`inference.py`、`visualize.py`、`models/tokenizer.py`、`models/transformer.py`、`retarget/optim_betas_smplx.py`、`retarget/retarget_motion_smplx.py`、`motion_lib/motion_lib_retarget.py`、`legged_gym/legged_gym/scripts/train.py`、`legged_gym/legged_gym/scripts/play.py`。
- **通读：** 已完成

## 目录结构与可运行入口核对

### 1. UH-1 文本生成与 MuJoCo 运动学可视化

- **`option.py` / `inference.py`：** 载入 CLIP `ViT-B/32` 文本编码器、`HumanoidVQVAE`（`models/tokenizer.py`，亦含 `VQVAE_34`）与 `UH1_Transformer`（`models/transformer.py`）；从 HuggingFace `USC-GVL/UH-1` 下载 `UH1_Action_Tokenizer.pth`、`UH1_Transformer.pth` 与统计量 `Mean.npy` / `Std.npy` 后，执行自回归采样 `trans_encoder.sample(feat_clip_text, False)` 并解码反归一化，保存 `[T, 34]` 关键点序列 `.npy` 到 `output/`（27 维 H1-2 DoF + 3 维根位置 + 4 维根四元数）。
- **`visualize.py`：** 使用 MuJoCo 渲染器加载 `./robots/h1_2/scene.xml`（注意：脚本内根四元数做 `[3, 0, 1, 2]` 重排以适配 MuJoCo `wxyz` 顺序，前 27 维为 `dof_pos`、`[27:30]` 为 `root_pos`、`[30:]` 为 `root_ori`），以 30 fps 导出 `.mp4` 预览。该步为纯运动学 `qpos` 回放，不含物理闭环控制。

### 2. 人到人形重定向流水线（`retarget/`）

- **`README-Humanoid-X.md` 文档状态核查：**
  - **Motion Retargeting：** 已完整给出脚本与示例配置，需用户自行按授权下载 SMPL v1.1.0 / SMPL-X v1.1 与 `PoseLib`；
  - **Video Clip Extraction / 3D Human Pose Estimation / Video Captioning：** `README-Humanoid-X.md` 均显式标注为 **"Work in progress"**，仅列出外部参考仓（VIBE、4DHumans、Video-LLaMA / VideoLLaMA 2），当前仓库不含从互联网原片端到端切段、打标、估计 SMPL 的脚本。
- **`retarget/optim_betas_smplx.py`：** 在 MuJoCo 中加载 `./humanoid_model/h1_2/scene.xml` 采集 Unitree H1-2 标准 T 姿下 12 个共有关键点（`keypoint_index = [4, 5, 6, 10, 11, 12, 17, 20, 22, 26, 29, 31]`，覆盖双髋/膝/踝/肩/肘/腕），优化 SMPL-X 体型参数 `betas`（受 `RANGE_BETAS = 5` 限制）并导出如 `betas_param/betas_param_r5_smplx.npy`。
- **`retarget/retarget_motion_smplx.py`：**
  - `process_file(...)` 读取原始 SMPL-X `.npz` 动作序列并下采样到目标帧率（`__main__` 中设为 `target_fps=60`），将坐标系旋转对齐至机器人坐标并抽取 SMPL-X `[1,4,7, 2,5,8, 16,18,20, 17,19,21]` 12 关键点轨迹；
  - `H1mRetargetKeypoint.retarget()` 使用 `torch.optim.Adam([joint_pos, root_trans], lr=0.005)` 结合 `Humanoid_Batch_H1` 前向运动学最小化 L1 关键点距离损失 + 二阶差分时间平滑损失（`lamb=0.05`），输出包含 `fps`、`root_pos`、`root_rot`、`dof_pos` 的 `.npy` 文件到 `./retarget_output/`（脚本内另注释保留了包含 `root_vel`、`keypoint_trans` 的 `forward_trackable` 完整导出分支）。

### 3. Isaac Gym 目标条件跟踪 RL 策略（`motion_lib/` & `legged_gym/`）

- **`motion_lib/motion_lib_retarget.py`：** 定义 `MotionLibRetarget`，读取 `.pkl` 或由重定向 `.npy` 目录自动生成并缓存 `motion_data_cmu_sample.pkl`，记录 27 个 H1-2 无手 DoF（每条腿 6 DoF、躯干 1 DoF、每条臂 7 DoF）的 `root_pos`、`root_rot`、`dof_pos`、速度差分及 12 关键点轨迹。
- **`legged_gym/legged_gym/scripts/train.py` & `play.py`：** 通过 `task_registry` 启动 `--task h1_2_mimic`（训练入口调用 `ppo_runner.learn` 驱动 `rsl_rl` 训练跟踪策略，示例数据为作者提供的 CMU MoCap 重定向样本而非全量 163.8K 视频数据）。

## 知识库映射

- [Humanoid-X（BFM 行为数据）](../../wiki/entities/dataset-bfm-humanoid-x.md)
- [arXiv:2412.14172 原文通读归档](../papers/humanoid_x_uh1_arxiv_2412_14172.md)
