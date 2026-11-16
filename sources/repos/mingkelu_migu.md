# MingkeLu/MIGU 官方源码仓库核查

- **仓库地址：** <https://github.com/MingkeLu/MIGU>
- **关联论文：** <https://arxiv.org/abs/2609.24995>（arXiv:2609.24995）
- **项目主页：** <https://multimodal-instruction.github.io/>
- **开源状态：** 部分开源（离线核心多模态 grounding 管线、100 帧真实 RGB-D 数据集 `captures/`、基线/消融脚本与 ROS Noetic 在线接口已开源；单轮 POMDP 澄清对话模块、ASR 与下游规划器未开源）
- **通读：** 已完成
- **核对范围：** 已核查 `MingkeLu/MIGU` 仓库目录树 `_tree.txt`（L1–504）、`README.md`（L1–135）、`EXPERIMENTS.md`（L1–288）、`pyproject.toml`（L1–49）、`config.yaml`（L1–26）、`migu/{__main__,pipeline,gesture_posterior,gesture,vlm}.py` 与 `planning_interface/{grounding_node,grounding_tool,gesture_monitor}.py`，并检索 `migu/` 与 `planning_interface/` 源码边界。

---

## 仓库内容与可运行入口

- **包管理与 CLI（`pyproject.toml` L1–49）：** 使用 `uv sync`（可选 `--extra ros`）安装；注册 6 个脚本入口：
  - 离线基准与分析：`migu`（`migu.pipeline:main`）、`migu-result-analysis`
  - ROS Noetic 在线接口：`migu-gesture-monitor`（`planning_interface.gesture_monitor:main`）、`migu-grounding-node`（`planning_interface.grounding_node:main`）、`migu-publish-instruction`、`migu-bag-demo`
- **基准数据与评估基线：**
  - `captures/` 完整包含 16 组场景/抓放（pick/place）任务的真实 RGB-D 图像、`point.json` 与 `metadata.json`（对应论文 100 场景基准）。
  - `baselines/` 提供 `geometry`、`POMDP`（定位基线）、`GIRAF`、`VLM-Grid`、`VLM-PIVOT`、`VLM-SoM`；`ablations/` 提供 `without-semantics`、`without-geometry`、`ray-distance`、`ray-angle`、`distance-threshold`、`semantic-owlv2`、`semantic-som`、`semantic-grid`、`semantic-pivot`。
  - `tests/` 包含单元测试。

---

## 运行时管线核对

1. **离线批处理（`migu/pipeline.py`, `config.yaml`）：**
   - 默认 VLM 为 `gpt-5.6-sol`（OpenAI Responses structured output `client.responses.parse` 返回 `BoundingBox` JSON，`bbox_count: 5`），SAM 服务默认请求 `http://127.0.0.1:4001`。
   - `batch_preparation` 读取样本后由 `generate_key_frames` 调用 MediaPipe Hands 提取 2D/3D 食指关键点（landmarks 5–8）；`ThreadPoolExecutor(max_workers=2)` 并行运行 `run_gesture`（`migu/gesture.py` -> `pixel_posterior_from_gesture` 计算球面 $\mathbb{S}^2$ 几何后验）与 `run_vlm_sam`（`migu/vlm.py` + `migu/sam.py`），再经 `fuse_pixel_wise` 按 `masks * posterior * confidence` 融合并按掩码 `.max(axis=(1,2))` 聚合归一化。
2. **ROS Noetic 在线接口（`planning_interface/`）：**
   - `gesture_monitor.py` 与 `grounding_node.py` 订阅 `/head_camera/rgb/image_raw`、`/head_camera/depth_registered/image`、`/usr_instruction`、`/ray_queries`，调用 `GroundingTool` 将带概率分布与 3D 信息的接地结果以 JSON 发布至 `/grounding_results`。

---

## 论文与源码差异核查

1. **外部依赖服务：** 运行时需额外启动外部仓库 `threefruits/perception_services`（`sam-server` 监听 4001 端口；OWLv2 消融另需 `owl-server` 监听 4000 端口）以及配置 `OPENAI_API_KEY` 调用云端 VLM API。
2. **参数默认值差异：** 论文 §IV-B 给出文献先验 $\sigma_u = 0.044\text{ rad}$ [35]、$\alpha_d = 0.04$ [36]；仓库实际默认值（`migu/gesture.py` L27–29 与 `planning_interface/grounding_tool.py` L123–126）设为 `sigma_u = 0.009`、`alpha_depth = 0.2`、`fitting_covariance_beta = 1.5`。
3. **未包含模块边界：** 检索 `migu/` 与 `planning_interface/` 源码确认，仓库仅覆盖多模态 grounding（输出候选分布），**不包含**论文实验中的 Whisper ASR、**单轮 POMDP 二元属性澄清对话模块**、Autolife S2 的 `cuRoboV2 + ContactGraspNet + VAMP` 规划器以及 Fetch 的 `PDDLStream + OWLv2` TAMP 规划器。
