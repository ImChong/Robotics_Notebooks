# 3DAgentWorld/HorizonStream — 官方推理源码核验

- **类型：** 官方代码仓库 / streaming 3D reconstruction
- **地址：** <https://github.com/3DAgentWorld/HorizonStream>
- **论文：** <https://arxiv.org/html/2605.23889v2>
- **项目：** <https://3dagentworld.github.io/horizonstream/>
- **权重：** <https://huggingface.co/NicolasCC/HorizonStream>
- **核验日期：** 2026-11-10
- **许可证：** LICENSE 声明 CC BY-NC-SA 4.0（非商业、署名、相同方式共享）；须自行核对依赖的各自授权。
- **开放范围：** README Release Status 已勾选权重、推理、评估、可选 loop closure、Viser/Gradio 演示、深度融合后处理与转换脚本；**未提供训练代码**（README 明确是公司政策限制）。
- **核对范围：** README、LICENSE、`infer.py`、`run_pipeline.py`、`horizonstream/core/{infer.py,model.py}`、`horizonstream/models/horizonstream.py`、`horizonstream/runtime/models/horizonstream.py` 的 GLACache 与 aggregator 执行路径、`horizonstream/loop/online_loop_reinfer.py` 的 loop 后处理入口。

## README 可运行入口

```bash
python scripts/download_weights.py
python infer.py --config configs/horizonstream_infer.yaml \
  --video-path /path/to/input.mp4 \
  --checkpoint checkpoints/HorizonStream.pt \
  --output-root outputs_horizonstream/input_video
# 不需要 loop closure 时加 --no-loop；长序列可加 --offload-outputs-to-cpu
```

准备好的 `generalizable` 数据可以换成 `--img-path`；`run_pipeline.py` 组合 inference、可选 loop 和有 GT 的 trajectory evaluation；还提供 `demo_gradio_viser.py`、`demo_viser_only.py`。

## 从代码逐项查到的真实执行路径

1. `infer.py:main` 解析 `--config/--video-path/--checkpoint` 等参数，`prepare_video_input_if_needed` 使视频格式进入 data config；在 `--no-loop` 未启用时预检查可选回环资产，随后调用 `horizonstream.core.infer.run_inference_cfg`，再按条件执行 `run_loop_closure_cfg`。
2. `run_inference_cfg` 初始化 `HorizonStreamModel` 加载 checkpoint、`HorizonStreamDataLoader` 加载序列；`_chunk_schedule` 将帧划成首窗及后续 sliding chunks。对每条序列只生成一套 `state = model.build_sequence_state()`。
3. 每个 chunk：调用 `model.forward_chunk(chunk_images, window_size, chunk_idx, state)`；包装层将 `frame_kv_caches/global_kv_caches/gla_cache` 转发至 `HorizonStream.forward_chunk`。
4. 模型内部 `HorizonStreamAggregator` 结合局部 attention 与 `_run_gla_global_block`，GLA 使用 `GLACache` 的 `state_ori/state_new`；`BatchCameraHead` 读 pose tokens，`metric_readout_head` 用 exp 输出尺度并修正相机平移和 DPT 深度，`DPTHead` 输出深度及置信度。
5. 每个 chunk 输出 `chunk_cam_map / depth / depth_conf`，核心推理层再调用 `model.advance_sequence_state`，`GLACache.advance` 将 `state_new` 变成下一 chunk 的 `state_ori`；末段清空。注意局部短窗 KV 与 GLA 的跨窗有界状态不可混为“所有 KV 永不重置”。
6. `run_inference_cfg` 对相机结果调用 `compute_motion_averaged_camera_maps`，写入 `poses/abs_pose.txt`、`poses/intri.txt`、`depth/dpt/*.npy` 与 `points/full.ply` 等；可选 offline pose 变体与 sky/point filtering 取决于 config。
7. 若启用且资产具备，`run_loop_closure_cfg` 经 `run_online_loop_reinfer` 做候选回环、重推理和位姿约束优化；loop 输出（如 `loop_abs_pose.txt` / `full_lc.ply`）是**另一个评估版本**。

## 工程边界和安全提示

- README 指出论文报告时 `--sliding-size 21`；`--sliding-size 1` 的纯逐帧情形约 8.5 GB peak GPU memory（官方测试环境），长序列建议 `--offload-outputs-to-cpu`。积累深度/点云/保存文件仍随帧数增长；不能对整个 Python 运行时声称严格 O(1)。
- `infer.py` 默认**尝试 loop closure**，用 `--no-loop` 才是纯在线推理；有些 loop 资产或第三方依赖需单独下载。论文 FPS 和 ATE 对齐实验设定时要固定 `--sliding-size`、LC 开关与 camera preprocess。
- README 的完整训练配置、训练数据和权重生成链条未作为可执行训练代码公开；无法仅凭此仓库独立重训或复现作者所有统计表。
- 本次为源码静态阅读，不包含 checkpoint 下载、GPU 推理、数据集评估或端到端复现实验。

## 关联

- [论文精读归档](../papers/horizonstream_arxiv_2605_23889.md)
- [项目页](../sites/horizonstream-project.md)
- [Wiki 实体](../../wiki/entities/paper-horizonstream.md)
