# 在宇树 G1 上实战强化学习 BeyondMimic，31 个动作，一个策略全学会（完整复现）

> 来源归档（blog / 微信公众号）

- **标题：** 在宇树 G1 上实战强化学习 BeyondMimic，31 个动作，一个策略全学会（完整复现）
- **类型：** blog（实操教程）
- **来源账号：** 深蓝学院-具身君 / 深蓝具身智能（文中称项目实现基于深蓝学院课程《人形机器人运动控制》）
- **原始链接：** https://mp.weixin.qq.com/s/dySSDOWHM1an8o_zm-fSSg
- **发表日期：** 2026-11-06
- **入库日期：** 2026-11-06
- **抓取方式：** `scripts/fetch_wechat_article.py`
- **一句话说明：** 把短视频里的舞蹈（狗熊哆嗦毛、闪身步、浪子踢球等）搬到 Unitree G1：GVHMR 从视频恢复人体全局运动 → GMR 重定向 → CSV → NPZ → 用多参考动作扩展的 BeyondMimic 风格追踪策略训练，一个策略学会 31 个动作。
- **说明：** 没有论文清单；涉及的项目库内均已有节点（GVHMR、GMR、BeyondMimic、unitree_rl_mjlab、mjlab），故不新建项目节点，新建一页实操 Query 并补回链。文末课程宣传（开课时间、主讲人）不收。

## 文章要点（作者表述）

- **选视频：** 全程要能看到全身、动作完整、画面只有一个人、背景简洁、光照均匀；遮挡或多人交叉会让姿态恢复明显劣化，误差一路传到训练。
- **GVHMR：** 称其为「浙江大学开源的从单目 RGB 视频恢复全局人体运动」，输出 `hmr4d_results.pt`（SMPLX 参数：全局平移、朝向、身体与手部姿态）；命令 `python tools/demo/demo.py --video <视频> -s`，称 `-s` 表示保存结果。50 系显卡需按 GVHMR 的 Blackwell 支持 issue 调整环境。
- **GMR：** 用「Retargeting from GVHMR to Robot」：`python scripts/gvhmr_to_robot.py --gvhmr_pred_file <pt> --robot unitree_g1 --record_video --save_path <路径>`；再 `python scripts/batch_gmr_pkl_to_csv.py --folder <目录>` 转成 CSV（每帧：root 位置 xyz + root 四元数 xyzw + 关节角）。
- **NPZ：** `csv_to_npz.py --input_file ... --output_name ... --input_fps 30 --output_fps 50 --robot g1`；文章称该脚本把帧率 30 → 50 fps（dt = 0.005 s、decimation = 4），在 Isaac Sim 中做正向运动学得到各刚体世界系位置 / 姿态 / 线速度 / 角速度，并上传 WandB Registry；NPZ 字段：fps、joint_pos、joint_vel、body_pos_w、body_quat_w、body_lin_vel_w、body_ang_vel_w。
- **多参考动作扩展（作者自己加的）：** 官方 unitree_rl_mjlab 与 whole_body_tracking 都只支持单个动作；作者加了 `MultiMotionCommands` 与 `train_multi.py`，给每个动作分配权重，按权重分配环境数（例：权重 1、1、2，4096 个环境 → 1024 / 1024 / 2048）。
- **训练：** 31 个动作，较难的（狗熊哆嗦毛、浪子踢球）权重 2，其余 1；`python scripts/train_multi.py Unitree-G1-Tracking-No-State-Estimation-Multi-Priors ... --num-envs 13312`；约 20000 轮，RTX 5080 约 23 小时，最终 mean_reward 约 32（作者表述，未核实）。播放建议用 viser 后端。
- **结论：** 一条链路对应三个核心问题：动作从哪来（数据与重定向）、怎么学得会（追踪策略训练）、怎么站得住（物理仿真与迁移）。

## 一手资料核对（我做的）

浅克隆四个官方仓库，只读 README 与脚本，未运行任何代码（提交：GVHMR `ee960bb`、GMR `25f56db`、unitree_rl_mjlab `1425b15`、whole_body_tracking `9db9158`）。**文章与一手资料不一致的地方：**

1. **GVHMR 的名称：** 文章写「Global-View Human Motion Recovery」；arXiv:2409.06662 题名为 *World-Grounded Human Motion Recovery via Gravity-View Coordinates*（2024-09-10）。
2. **`-s` 的含义：** `tools/demo/demo.py` 里 `-s/--static_cam` 的帮助是 "If true, skip DPVO"（静态相机，跳过 DPVO 相机运动估计），README 示例 `demo.py --video=... -s` 同义；不是「保存结果」。结果文件 `hmr4d_results.pt` 由默认流程写出（代码里先检查该文件是否存在再计算）。
3. **GMR 保存格式与扩展名：** `gvhmr_to_robot.py` 用 `pickle.dump` 写 `--save_path` 指定的路径，不看扩展名；文章示例却写 `.npz`。而 `batch_gmr_pkl_to_csv.py` 只处理扩展名为 `.pkl` 的文件（其他 `continue`），所以示例的 `.npz` 文件会被跳过（静态阅读的推断）。该脚本还把 > 30 fps 的动作降到 30 fps，输出到 `<folder>/csv/*.csv`，列为 root_pos(3) + root_rot(4，GMR 代码注释「save from wxyz to xyzw」) + dof_pos，与文章描述一致。
4. **「官方库只支持单动作」成立：** unitree_rl_mjlab README 的训练命令是 `scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=<单个 npz>`，仓库内没有 `train_multi` / `MultiMotion` / `motion_weights`；多动作扩展是作者自己的代码，文章与官方仓库均未提供。
5. **BeyondMimic 官方库已退役：** whole_body_tracking README 写明「This repository is retired and no longer maintained」，BeyondMimic 追踪实现转到 mjlab（MuJoCo-Warp 的 Isaac Lab 风格 manager API）的 tracking task。文章同时引用了 whole_body_tracking 与 unitree_rl_mjlab。
6. **文章把两套工具链的描述混在一起（推测为沿用 BeyondMimic 文档的措辞）：** 文章说「基于 Isaac Lab / 在 Isaac Sim 中做正向运动学并上传 WandB Registry」，这符合 whole_body_tracking 的 `csv_to_npz.py`（用 Isaac Lab `AppLauncher`）；但文中实际训练用的是 unitree_rl_mjlab，其 `csv_to_npz.py` 用 MuJoCo 时间步长（`sim_cfg.mujoco.timestep`），脚本里没有 wandb / registry 字样。参数写法也不同：unitree_rl_mjlab README 用连字符（`--input-file`、`--output-name`），whole_body_tracking 用下划线（`--input_file`）。
7. **G1 自由度：** GMR 表中 `unitree_g1` 为 29 DoF（腿 2×6 + 腰 3 + 臂 2×7），unitree_rl_mjlab 另有 `Unitree-G1-23Dof-Tracking-No-State-Estimation`，与文章「23 或 29 个可控关节」一致。
8. **未核实：** 5080 上 23 小时、20000 轮、mean_reward ≈ 32；13312 个环境的显存占用；GVHMR 的 Blackwell issue 内容；多动作扩展的实现与效果（无代码、无真机）；演示视频。文章没有给真机结果。

## 对 wiki 的映射

- [视频舞蹈到 G1：GVHMR → GMR → BeyondMimic 多动作训练实操](../../wiki/queries/g1-video-dance-gvhmr-gmr-beyondmimic-multi-motion.md)
