# FaceFusion — 官方仓库与源码调用链

- **标题：** FaceFusion — Industry leading face manipulation platform / 多处理器人脸影像处理平台
- **类型：** repo / face manipulation / image and video processing
- **官方仓库：** https://github.com/facefusion/facefusion
- **官方文档：** https://docs.facefusion.io/
- **核对版本：** master @ decc6c9d8f8e014878624452c6e8fe208726b4ec（Patch 3.9.2；提交日期 2026-10-26）
- **入库日期：** 2026-11-08
- **代码开放状态：** 已公开完整可浏览的 Python 应用入口、工作流、处理器和测试；模型由运行时按配置下载，模型权利须另查。
- **许可证核验：** 根目录 LICENSE.md 只有「OpenRAIL-AS license」和 Copyright (c) 2026 Henry Ruhs；README 徽章亦显示 OpenRAIL-AS。当前许可证文件没有展开完整授权条款，不能据此推断任意商业用途已获许可。
- **核对范围：** 已通读官方 README、安装说明及处理器列表；阅读 facefusion.py、facefusion/core.py 的 CLI/调度完整主链，facefusion/program.py 的参数组织与处理器注册，facefusion/processors/core.py、facefusion/workflows/image_to_image.py、image_to_video.py、to_image.py、to_video.py 的主处理路径，以及 face_swapper/core.py 的预检查、身份提取、裁剪、推理、掩膜回贴及 process_frame，核对 face_selector.py 和 face_enhancer/core.py 的相关函数；未逐行审计全部处理器、依赖、预训练模型或作运行时性能实测。
- **通读：** 已完成

## 入口与可复核路径

1. [README — 安装入口、CLI 命令及文档](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/README.md)
2. [facefusion.py — conda.setup() → core.cli()](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion.py)
3. [core.py — route() / process_headless() / process_step() / conditional_process()](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/core.py)
4. [program.py — create_program() / create_processors_program()](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/program.py)
5. [processors/core.py — 动态加载与处理器接口约束](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/processors/core.py)
6. [workflows/image_to_image.py — 图片任务顺序](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/workflows/image_to_image.py)
7. [workflows/image_to_video.py — 磁盘帧/内存帧分支](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/workflows/image_to_video.py)
8. [workflows/to_image.py — process_temp_frame() 与 finalize_image()](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/workflows/to_image.py)
9. [workflows/to_video.py — extract/process/merge/restore 路径](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/workflows/to_video.py)
10. [face_swapper/core.py — 人脸选取、预处理、模型运行、粘贴回图](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/processors/modules/face_swapper/core.py)
11. [face_selector.py — 选择与筛选目标人脸](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/facefusion/face_selector.py)
12. [LICENSE.md — 许可证标识原文](https://github.com/facefusion/facefusion/blob/decc6c9d8f8e014878624452c6e8fe208726b4ec/LICENSE.md)

## 源码机制核验

- 入口 `facefusion.py` 调用 `conda.setup()` 和 `core.cli()`；`core.route()` 将 `run` 交给 UI，将 `headless-run`、`batch-run`、`job-*` 交给任务执行器/管理器。
- `process_headless()` 建立 job、step、提交并运行；`process_step()` 写入工作状态；`conditional_process()` 根据目标文件识别图像或视频工作流、检查选中处理器，再分派。
- 图像：`analyse_image` → `prepare_image` → `process_image`（调用 `process_temp_frame`）→ `finalize_image`。
- 视频：`analyse_video` → `setup` → 若 strategy=disk 则 `extract_frames` / `process_disk_frames` / `merge_frames`；若 strategy=memory 则 `process_memory_frames`；末尾 `restore_audio` / `finalize_video`。
- `processors/core.py` 用 `importlib` 动态加载 `facefusion.processors.modules.<name>.core`，要求各处理器实现预检查、参数注册、前后处理及 `process_frame` 接口。
- `face_swapper/core.py`：读取源图并校验源脸 → `get_static_faces`、`extract_source_face` → `select_faces` 定位目标 → `warp_face_by_face_landmark_5` 对齐 → 可选 box/occlusion/area/region mask → `forward_swap_face` 用推理池调用 ONNX 会话 → `paste_back` 贴回输出帧。不同模型类型可走源图或 embedding 表征，不能将全部实现概括为单一 Inswapper。
- 代码中 `content_analyser` 位于图像/视频分析路径，但代码门控不等于对授权、冒充或真实性的充分合规保证。

## 工程边界与复现

- 文档提供 Conda / Python 3.12 安装路径，包含 CPU、CUDA 12、CoreML、DirectML、MIGraphX、OpenVINO 等安装选择；具体版本必须按官方平台说明配套。
- 处理器可组合，官方 CLI 文档列出 face_swapper、face_enhancer、face_editor、expression_restorer、lip_syncer 等；源码目录还包含不同模块，不能不核实版本便宣称所有处理器在所有安装环境均可运行。
- 运行时预检查需要 curl、ffmpeg、ffprobe，模型下载和具体加速后端还受网络、设备和依赖限制。**本次未运行软件，不提供帧率、精度或显存保证。**
- 面向数字人演示或 HRI 的可借鉴点在于「检测/识别—帧处理—推理—合成—媒体输出」流水线；它不输出机器人控制策略、3D 运动学或已验证的真实人物意图。

## 对 wiki 的映射

- [FaceFusion 项目实体](../../wiki/entities/facefusion.md)
- [官方文档/项目主页核对](../sites/facefusion-official.md)
- [Deep-Live-Cam 比较节点](../../wiki/entities/deep-live-cam.md)
