# RavenLeeANU / Gen2Humanoid

> 来源归档

- **标题：** Gen2Humanoid
- **类型：** repo
- **维护者（GitHub 显示）：** RavenLeeANU（Wenrui）
- **链接：** https://github.com/RavenLeeANU/Gen2Humanoid
- **默认分支：** `master`
- **Hugging Face 数据集：** https://huggingface.co/datasets/RavenLeeANU/Gen2Humanoid-HY-Motion-1.0
- **许可：** 核心 `g2h` 包 **MIT**；端到端使用因依赖 **HY-Motion-1.0** 而受 **Non-Commercial Scientific Research Use Only** 约束（见仓库 README License 节）
- **入库日期：** 2026-06-15
- **一句话说明：** 文本提示 → **HY-Motion-1.0** 人体运动 → **SMPL-X** 格式转换 → **GMR** 人形重定向 → **viser** 人机并排可视化的端到端开源管线；子模块打包官方 HY-Motion 与 GMR，开箱支持 Unitree G1 / H1、Booster T1 等。
- **沉淀到 wiki：** 是 → [`wiki/entities/gen2humanoid.md`](../../wiki/entities/gen2humanoid.md)

## 核心摘录（据 README，2026-06-15）

1. **主干流水线** — `[Text Prompt] → HY-Motion → SMPL-X → GMR → [Humanoid Robot Motion]`；CLI 入口 `scripts/pipeline.py`，一键脚本 `commands/run_pipeline.sh`。
2. **子模块依赖** — `third_party/HY-Motion-1.0`（腾讯混元文本→人体运动）、`third_party/GMR`（通用运动学重定向）；需 `git clone --recursive` 或 `git submodule update --init --recursive`。
3. **环境与权重** — Conda `python==3.10`；HY 预训练权重经 `commands/download_hy_model.sh`；SMPL-X 经 `commands/download_smplx.sh`；GMR 以 `pip install -e .` 安装。
4. **默认机型** — README 表列 `unitree_g1`（29 DoF）、`unitree_h1`（19 DoF）、`booster_t1`（23 DoF）；完整列表见 GMR 子模块 README。
5. **输出格式** — `robot_motion.pkl`：`fps`、`robot_type`、`num_frames`、`root_pos`、`root_rot`（四元数 xyzw）、`dof_pos`。
6. **可视化** — `g2h/visualise/` 下 `robot_viser.py` / `smplx_viser.py` / `motion_player.py`；部分 viser 代码致谢 [video2robot](https://github.com/AIM-Intelligence/video2robot)。
7. **路线图（TODO）** — 后处理（脚滑/自碰修补）、条件生成（姿态/轨迹约束）、动作混合、下游 motion tracking 模块尚未实现。

## 对 wiki 的映射

- **实体页**：[`wiki/entities/gen2humanoid.md`](../../wiki/entities/gen2humanoid.md) — 「文本→人形机器人参考」轻量工程胶水层，串联已有 HY-Motion 与 GMR 能力。
- **上游方法**：[`wiki/methods/hy-motion-1.md`](../../wiki/methods/hy-motion-1.md)（T2M 生成）、[`wiki/methods/motion-retargeting-gmr.md`](../../wiki/methods/motion-retargeting-gmr.md)（几何重定向）。
- **概念交叉**：[`wiki/concepts/motion-retargeting-pipeline.md`](../../wiki/concepts/motion-retargeting-pipeline.md) — 生成式上游 + 运动学前端的端到端实例。
- **相邻仓库**：[`sources/repos/tencent_hunyuan_hy_motion_1_0.md`](tencent_hunyuan_hy_motion_1_0.md)（HY 官方仓）、[`sources/repos/soma_retargeter.md`](soma_retargeter.md) / [`sources/repos/kimodo.md`](kimodo.md)（其他重定向前端对照）。

## 备注（维护者）

- README 中 `convert_smpl.py` 负责 HY 输出到 **SMPL-X** 的格式桥接；进入 GMR 前须确认坐标系与关节定义与 GMR 子模块期望一致。
- **商用前**须分别复核 HY-Motion-1.0 与 GMR 许可；本仓 MIT 不覆盖子模块限制。
- 仓库约 **140+ stars**（2026-01 最后提交）；无独立论文页，以 GitHub README 与 Demo 表为一手真值。
