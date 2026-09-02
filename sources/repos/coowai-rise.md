# COOWAI/RISE

> 来源归档

- **标题：** RISE: Adaptive Imagination for World Action Models
- **类型：** repo
- **来源：** 酷哇科技（COWARobot）/ 上海交通大学 / 河海大学
- **链接：** <https://github.com/COOWAI/RISE>
- **主页：** <https://cowarobot-ai.github.io/RISE/>
- **论文：** <https://arxiv.org/abs/2608.20430>
- **许可：** MIT
- **入库日期：** 2026-09-02
- **一句话说明：** 驾驶 WAM 的自适应 latent 想象实现：冻结 V-JEPA 风格编码器 + Predictor + 扩散 Planner + 手动七段控制器链；公开代码与配置，**不公开权重**。
- **沉淀到 wiki：** [`wiki/entities/paper-rise-adaptive-imagination-wam.md`](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)

---

## 开源边界（2026-09-02 核查）

README 仍写「论文 / 数值 / 反事实数据 / 权重未发布」。同期项目页与 arXiv **已发布论文**，HF **已放 CounterDrive tar**。仓库本身：

| 有 | 无 |
|----|----|
| 训练/评测代码、七份扁平 YAML、无数据结构测试、中英复现指南 | 官方 checkpoint |
| `app/vjepa_cowa_world_model/`、`src/`（V-JEPA 模型与工具） | NAVSIM / NuPlan / metric-cache 资产 |
| Oracle CLI、NavTest EPDMS 封装 | 自动工作流引擎（全程手动 handoff） |

完整训练必须自填所有 `/path/to/`；未改路径会在生产 preflight 失败，代码不搜索替代资产。

---

## 仓库入口（对齐 `docs/reproduction.md`）

| 组件 | 说明 |
|------|------|
| 安装 | Python ≥3.11；`pip install -e .`；`export PYTHONPATH="$(pwd):${PYTHONPATH:-}"` |
| 冷导入 | `import app.vjepa_cowa_world_model; import src` |
| Predictor | `torchrun … -m app.main --fname configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml --train-script train_latent_predictor` |
| P0 / P1 | `--train-script train_predictor_rollout_planner`（`02_p0_uniform.yaml` / `05_p1_full.yaml`）；checkpoint **人工**拷到 `handoff/p0_selected.pt` / `p1_selected.pt` |
| Field / Calibration / Stop / Gate | `--train-script train_cvoi_offline`（`03`–`04`、`06`–`07` YAML） |
| CF sidecar | `tools/generate_navsim_cf_trajectory_quality.py` |
| Oracle | `tools/run_cvoi_manual_oracle.py`：`build-manifest` → `score --horizon {0..4}` → `aggregate` |
| EPDMS | `tools/run_cvoi_direct_epdms.py --config …/full_controller.yaml`（在线选 H0–H4，不暴露强制 horizon） |
| 无数据检查 | `pytest tests/test_cvoi_manual_full_configs.py tests/test_cvoi_direct_epdms_config.py` |

控制器链（README / 复现指南原句）：**`P0 → Field → Calibration → P1 → Stop → Oracle → Gate`**，再跑 NavTest Full-controller EPDMS。Predictor YAML 属七配置之一，但其输出 **不** 作为 P0 父节点。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-rise-adaptive-imagination-wam](../../wiki/entities/paper-rise-adaptive-imagination-wam.md) | 论文实体与评测读法 |
| [paper-vjepa2](../../wiki/entities/paper-vjepa2.md) | 冻结视觉编码器谱系 |
| [OpenDriveLab RISE](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) | **同名不同文**：操作想象 RL，不是本仓 |
