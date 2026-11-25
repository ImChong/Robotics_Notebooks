# NavProbe 官方代码

- **类型：** repo
- **链接：** https://github.com/liujy25/NavProbe
- **论文：** https://arxiv.org/abs/2609.27526
- **归档：** [论文原文](../papers/navprobe_arxiv_2609_27526.md)
- **许可证：** MIT；第三方素材各自保留许可。
- **核对日期：** 2026-11-25
- **开放状态：** 公开仿真评测实现与运行入口；场景数据需按 Matterport3D/HM3D 的授权流程取得，外部 VLM 服务及检测器权重需另配；未确认完整真机部署实现。

## 复现入口

Python 3.9、Habitat 0.3.3，按 README 安装 `.[llm,habitat,detectors]`，配置模型 API 和检测器。论文用 GPT-5.5 与 YOLO-World；当前 README 默认检测器为 GroundingDINO，不能直接等同论文配置。

- `python scripts/run_vlnce_agent.py --dataset r2r` 或 `--dataset rxr`
- `python scripts/run_hm3d_agent_batch.py --dataset hm3dv2`
- 可选 LightNav 垂直导航需另启服务。输出在 `runs/<dataset>/`。

## 已静态核对的调用链

- [scripts/run_vlnce_agent.py](https://github.com/liujy25/NavProbe/blob/main/scripts/run_vlnce_agent.py)：转发到 runner.main。
- [navprobe/runners/vlnce_runner.py](https://github.com/liujy25/NavProbe/blob/main/navprobe/runners/vlnce_runner.py)：构建 episode 列表、run_batch、VLNCEEnv、run_navprobe_agent_episode。
- [navprobe/agent/runner.py](https://github.com/liujy25/NavProbe/blob/main/navprobe/agent/runner.py)：reset、构造 Graph/RuntimeCache/LLMClient/LocalmapActionExecutor、refresh_state、决策、record_step 和 finalize_run。
- [navprobe/agent/visual_step_execution.py](https://github.com/liujy25/NavProbe/blob/main/navprobe/agent/visual_step_execution.py)：plan_visual_navigation_action 后按 stop、backtrack、vertical 或普通 waypoint 分派执行；终点接近后保存待验证状态。

以上为静态源码阅读，未安装依赖、调用外部模型或运行 Habitat。

[知识节点](../../wiki/entities/paper-agentic-2609-27526.md)
