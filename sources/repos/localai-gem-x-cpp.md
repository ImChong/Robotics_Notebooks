# LocalAI gem-x.cpp（C++ / GGML 本地推理实现）

> 来源归档（ingest）

- **类型：** repo / deployment implementation
- **仓库：** <https://github.com/localai-org/gem-x.cpp>
- **项目说明：** 实时摄像头或离线录制视频转为 SOMA 77 关节骨架；C++23 / GGML 移植自 NVIDIA GEM-X
- **上游模型 / 参考实现：** <https://github.com/NVlabs/GEM-X>
- **模型权重：** <https://huggingface.co/LocalAI-io/GEM-X-GGUF>
- **相关研究线：** [GENMO / GEM（NVIDIA）](../../wiki/methods/genmo.md)；该链接表明技术关系，不把移植声称为独立论文
- **许可：** gem-x.cpp 原生代码贡献 Apache-2.0；转换模型 NVIDIA Open Model License；DINOv3、MHR、SAM 3D Body 和其他第三方组件另有条款。参阅仓库 docs/LICENSING.md、LICENSES/ 与模型卡。
- **核查日期：** 2026-10-08
- **Wiki 主节点：** [gem-x.cpp（GEM-X 的 GGML/C++ 本地实现）](../../wiki/entities/gem-x-cpp.md)

## 一句话摘要

LocalAI 社区将 GEM-X 的观测模型和时序回归管线转换为 GGUF，并用 C++ / GGML 与 Vulkan 或 CPU 后端运行；浏览器 demo 提供实时骨架预览和最多 120 采样帧的离线视频骨架 GLB 导出。

## 仓库功能与运行入口

- **实时摄像头：** YOLOX-X 检测 + ViTPose 产生 77 个 2D 关键点，再进入 GEM-X 的 30 帧 rolling window；两帧 warm-up 后输出最新 pose。直播默认每 5 帧检测一次，可调到每帧。此分支不需要 SAM 3D Body。
- **离线视频：** 视频经过 YOLOX / ByteTrack 与 ViTPose；SAM 3D Body 提供 GEM-X 所需 pose feature，完整片段回归后可执行 contact correction、grounding、IK，并导出带同步视频的骨架视图及动画 GLB。当前浏览器 demo 限 120 个采样帧；GLB 为骨架动画，不是完整 body mesh。
- **模型下载：** README 要求 gem-x-contact-f32.gguf、vitpose-f32.gguf、yolox-f32.gguf；离线另需 SAM 3D Body backbone、pose branch 与 MHR 文件。模型资产不包含在 Git 仓库或 GEM-X GGUF 下载包中。
- **构建：** Linux 是 README 的 tested platform。依赖包括 C++23、CMake、Ninja；浏览器 demo 需要 Go。GPU 路线用 Vulkan preset；README 另提供 CPU release preset 示例。
- **集成：** 原生 streaming API 可消费时间戳 SOMA / SMPL pose。robot control 与 LocalAI 网络 streaming adapter 是单独集成；demo 未宣称直接连某个机器人。
- **验证：** 官方仓库的 2026-09-18 strict-F32 parity 文档使用一段 72 帧视频，报告 native 与 upstream 接近但非 bit-identical 的关节输出；作者明确说明这不是多视频质量评测，也不验证机器人/相机延迟。
- **许可分层：** C++ 原生代码 Apache-2.0 不自动覆盖模型权重及第三方资产。模型卡将 GGUF 标为 NVIDIA Open Model License，并说明这是格式/计算图转换，非重训练或低比特量化；具体文件及衍生依赖须逐项核对。

## 对 Wiki 的映射

- [gem-x.cpp 实体节点](../../wiki/entities/gem-x-cpp.md)：运行路径、输入输出、构建和适用边界。
- [GENMO / GEM 方法页](../../wiki/methods/genmo.md)：描述上游研究关联，并链接到该社区维护的 C++ 移植。
