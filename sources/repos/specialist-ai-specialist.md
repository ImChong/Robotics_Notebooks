# specialist-ai/specialist — 采集数据管线与源码核对

- **类型：** repo / 一手源码
- **仓库：** https://github.com/specialist-ai/specialist
- **项目：** https://specialist-ai.github.io/
- **关联硬件：** https://github.com/specialist-ai/specialist_description
- **关联驱动：** https://github.com/specialist-ai/specialist_sdk 、 https://github.com/specialist-ai/feetech_driver 、 https://github.com/specialist-ai/insta360_driver
- **入库日期：** 2026-11-09
- **核对范围：** 已阅读主仓完整 README、scripts/convert.py、scripts/estimate.py、scripts/state.py 的入口与核心调用；核对 scripts/catalog.py 入口说明、官方组织 profile 和 specialist_sdk / feetech_driver / insta360_driver README。侧重可见的采集与离线重建链路；没有逐行审计所有 C++/GPU 优化文件，也没有下载真实视频、GPU 实跑或真机部署。
- **通读：** 已完成

## 真实代码路径

| 步骤 | 入口与数据 |
|---|---|
| 原始录制 | 每台 Insta360 一对 `.insv` 前/后鱼眼镜头；配合 IMU、固定锚点 AprilTag 与夹爪指节标签 |
| 1. Convert | `scripts/convert.py:main`，`load_insv → probe_capture → sync_two_captures → normalize_h264 → write_base_recording`，产出 `base/<session-id>.rrd` |
| 2. Estimate | `scripts/estimate.py:main`，`SessionRecording → discover_camera_tags → prepare_camera → cross_sightings → solve_joint → build_run_estimate → write_estimate_layers`，产出 `estimate/` 与 `viz/` Rerun 层；底层 VI-BA 使用 GTSAM + GPU |
| 3. State | `scripts/state.py` 从 base 和 estimate 生成时序化双手图像、世界系位置/旋转/速度、夹爪开合角。默认 `scale=8`、`f_row=100`，对应 `state_s8_r100` |
| 4. Catalog | `scripts/catalog.py` 将 state 目录登记至 Rerun catalog；**可选**，不是训练器 |
| 机器人端 | `specialist_sdk`、`feetech_driver`、`insta360_driver` 有可读源码与文档；整臂策略训练/推理路径仍待发布 |

## 官方 README 入口

```bash
uv sync
uv run python scripts/convert.py --input-insvs path/to/g0_00.insv path/to/g1_00.insv --output-dir logs
uv run python scripts/estimate.py --base logs/base/<session-id>.rrd --anchor.tag-ids <anchor-tag-id> --output-dir logs
uv run python scripts/state.py --base logs/base/<session-id>.rrd --output-dir logs
# 可选
uv run python scripts/catalog.py logs/state_s8_r100 --catalog-url rerun+http://host:port
```

上述为带占位符的官方命令格式，不可未经填入真实路径和 anchor ID 直接运行。 `estimate.py` 需要 Linux GPU / CUDA solve stack，`uv sync` 首次构建 C++17 MEI factor 并需要 Git/编译环境。

## 观测与动作含义

`state` 存储后视鱼眼缩放图像、双手相对世界锚点的位姿与线/角速度、手指开合角。训练器可从过去状态构建 observations、由未来状态构建 action targets；**目前没有开源训练器可据此验证具体 policy obs/action tensor layout**，也不能把“100Hz state row”理解为真实图像 100FPS 或整机控制 100Hz。

## 驱动许可与安全

- 主仓以仓库实际 LICENSE 为准；运行可能涉及第三方 SDK、Boost、GTSAM 等各自许可与二进制依赖。
- `insta360_driver` README 警示随包的厂商 `libCameraSDK.so` 不可作为独立公开 wheel 发布；项目源码开放不代表所有 SDK 可自由再分发。
- `specialist_sdk` 有舵机过热/通信故障释放扭矩、关节行程夹持，但 README 明示**没有 stall detection 与 command watchdog**，上机仍需独立硬件安全防护。
- 系统未附可由本次访问确认的端到端训练基准、公共 checkpoint 或真机成功率；不宣称实际复现。

## 对 wiki 的映射

- [Specialist 实体](../../wiki/entities/specialist-umi.md)
- [官方站点与硬件文档](../sites/specialist-ai.md)
