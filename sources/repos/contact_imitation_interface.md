# CIMI — ariszxxu/CIMI 官方源码与可复现性核验

- **类型：** repo
- **代码：** https://github.com/ariszxxu/CIMI
- **论文：** https://arxiv.org/html/2610.41028v1
- **项目页：** https://nus-lins-lab.github.io/CIMIWeb/
- **入库日期：** 2026-11-03；2026-11-09 核查更新
- **核对范围：** 阅读 README、pyproject.toml、docs/DATA.md / TRAINING.md / DEPLOYMENT.md / TASKS.md / RESOURCES.md / VALIDATION.md，检查 `src/data_preprocess/cli.py`、`pipeline.py`、`processing/recipe.py`、`src/policy/training/workspace.py`、`models/basepolicy/multi_finger_policy.py`、`src/deploy/cli.py`、`preview.py`、`inference/policy.py` 相关执行路径。仅核验上述关键源码和官方运行命令，**未逐行审计整仓，也未运行 GPU / 真机**。
- **通读：** 已完成
- **对 wiki 的映射：** [CIMI 唯一实体](../../wiki/entities/paper-contact-imitation-interface.md)

## 真实可运行入口及调用链

| 阶段 | 官方 CLI / Python 文件 | 核查到的动作 |
|---|---|---|
| 安装 | `pyproject.toml` | Python 3.11，暴露 `cimi-data`、`cimi-train`、`cimi-deploy`、`cimi-collect` |
| 数据处理 | `cimi-data process --task leap/coin ... --run` → `data_preprocess.cli:main` → `pipeline.py` 的 `process` → `processing.recipe:main` | 读取 MP4bundle/manifest，运行 pose → retarget → render → RADIO cache，校验 Zarr 与样本顺序；需原始录制及外部立体/视觉资源 |
| 接触面追踪 | `python -m data_preprocess.pose_estimation batch` | AprilCube/DeepTag 角点定位，输出接触面轨迹 |
| 重定向 | `python -m data_preprocess.retargeting` | 按 LEAP/Wuji 动力学和手型合同重定向 |
| 视角转换 | `cimi-data render leap-views` / `wuji-views` | 场景双目深度估计、重投影与机器人叠加 |
| 低层训练 | `python -m policy.training.workspace --config-name low_level low_level_setting=leap/three_finger_release` | `workspace.py` 与 Hydra 配置运行 decoder 训练并输出 `direct_decoder.pt`（灯泡使用三指） |
| 高层训练 | `python -m policy.training.workspace setting=leap/bulb/rel_frames_two_wrists_three_fe setting.frozen_decoder_checkpoint=/.../direct_decoder.pt` | `workspace.py` 从处理好的 Zarr / RADIO 特征构建 `RelativeFingertipFrameHierarchyFlowPolicy`，冻结匹配 decoder 训练规划器；请以实际 TASKS.md 对应设置名核对 |
| 纯离线预检 | `python scripts/release/validate.py smoke ...` | 严格重新加载所存策略权重，不代表论文成功率或真机成功 |
| 部署 | `python -m deploy preview --config local.yaml --checkpoint model.ckpt --preflight` | `deploy.cli:main` → `deploy.preview:main` → `preflight` → `InferencePolicy`；核查 checkpoint/decoder SHA、相机顺序、URDF、标定、初始姿态 |
| 交互执行 | 通过预检后移除 `--preflight` | 预览器进入监督模式，操作者明确授权初始化/执行；实时相机处理/关节反馈经 `InferencePolicy` 推理、执行安全约束后发送机器人命令，**启动预览不自动授权运动** |

注：低层三指正式配置存在 `leap/three_finger_release`；高层 bulb 配置实际名称以 `docs/TASKS.md` 和仓库配置为准。顺序图只展示在源码与 README 中核实的 CLI、模块与依赖，没有虚构万能的 `train.py` 或 `eval.py`。

## 官方基本操作（需本地数据/权重）

```bash
# 仅打印步骤/检查配置，不执行数据处理
cimi-data process --task leap/bulb

# 完整处理需原始录制、任务配置、相机参数、模型缓存
cimi-data process --task leap/bulb --config /absolute/path/local/processing.yaml --run

# 灯泡三指 decoder，随后用匹配 checkpoint 训练高层规划器
python -m policy.training.workspace --config-name low_level low_level_setting=leap/three_finger_release

# 部署前检查（不发机器人运动命令）
python -m deploy preview --config /absolute/path/local-preview.yaml --checkpoint /absolute/path/checkpoint.ckpt --preflight
```

## 开放程度与复现条件

- **公开**：预处理、动作重定向、RADIO 特征缓存、planner/decoder 训练、LEAP/Wuji 预览与部署源码、配置与测试，以及硬件装配说明。
- **未公开下载**：四套原始人类示教、六套训练 Zarr、已训练的策略和底层 decoder；历史训练所需旧数据/初始权重也未提供公开获得途径。详见 [RESOURCES.md](https://github.com/ariszxxu/CIMI/blob/main/docs/RESOURCES.md)。
- **测试证据与限制**：官方 [VALIDATION.md](https://github.com/ariszxxu/CIMI/blob/main/docs/VALIDATION.md) 报告 2026-10-09 源码检查离线 1624 passed / 7 skipped；此检查没有进行真实数据完整重训、真机 commissioning 或论文指标复核。本文没有独立运行或复测。
- **部署安全**：使用本机相机标定、任务对应动作/手型约定、decoder 权重 SHA、工作空间与避障守卫；不把参考样机校准当作新设备校准。

## 关联

- [论文归档](../papers/contact_imitation_interface_arxiv_2610_41028.md)
- [官方展示和开源边界](../sites/contact-imitation-interface.md)
- [CIMI 主 wiki 实体](../../wiki/entities/paper-contact-imitation-interface.md)
