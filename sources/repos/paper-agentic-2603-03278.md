# tether-research/tether（Tether 官方代码库）

- **仓库地址：** https://github.com/tether-research/tether
- **协议 / 分支：** MIT License / `main`
- **关联论文：** [Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping](paper-agentic-2603-03278.md)（arXiv:2603.03278, ICLR 2026）
- **核对边界：** 基于仓库元信息、文件目录抽样（`files_sample`）与 `README.md` 预览文本进行的静态代码结构核对，未在本地配置真机或 GPU 环境运行

## 目录结构与关键入口

- **配置与提示词：**
  - `conf/config.yaml`：配置 VLM API 密钥（`api_key_smart`、`api_key_fast`）等全局参数。
  - `conf/setting/real.yaml`：配置真机示范列表 `demo_names`、ZED 相机序列号及工作空间越界阈值 `oob_bounds`。
  - `prompts/plan_actions.txt`、`prompts/evaluate_success.txt`：VLM 任务规划与执行成功判定的提示词模板。
- **感知服务与核心算法模块：**
  - `serve_geo_aware.py`：启动 GeoAware-SC（2D 语义关键点对应）服务。
  - `serve_mast3r.py`：启动 MASt3R（跨视角几何一致性过滤）服务。
  - `extract_keypoint_trajectory.py` / `annotate_trajectory.py`：从原始演示中提取或标注关键路点与关键点轨迹摘要。
  - `run_correspondence.py`：调用对应模型生成目标场景关键点匹配与 3D 反投影筛选。
  - `warp_trajectory.py`：执行基于空间插值的轨迹扭曲与弧长速度重采样。
  - `query_gemini.py`：封装 Gemini VLM 规划与成功评估调用。
  - `ucb.py`：实现多臂赌博机 UCB 采样，用于动态筛选高成功率源演示。
- **执行入口与工具库：**
  - `runner.py`：主运行入口，支持单次滚出（`python runner.py mode=single action=...`）与自主玩耍循环（`python runner.py mode=cycle`），轨迹保存至 `data_real/runs/<run_name>/`。
  - `utils/`：`robot_utils.py`（对接 Eva Franka 底层，提供 `collect_scene_image()` 与 `send_trajectory()`）、`calibration_utils.py`、`geometry_utils.py`、`vlm_utils.py`、`statistics_tracker.py`、`annotation_utils.py`、`drawing_utils.py`、`timer_utils.py`、`misc_utils.py`。

## 运行依赖与复现边界

- 需配置三个独立 Conda 环境（Ubuntu 20.04 测试）：`tether`（Python 3.10 + `requirements.txt`）、`GeoAware-SC` 与 `MASt3R`。
- 依赖外部 Gemini API 密钥及配套机器人底层控制栈 [Eva](https://github.com/willjhliang/eva)（基于 DROID 软件栈的 Franka 控制服务）与双 ZED 相机标定数据。
