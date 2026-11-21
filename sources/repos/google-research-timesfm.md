# google-research/timesfm（TimesFM 3.0）

> 官方源码归档

- **标题：** TimesFM — Google Research 时间序列基础模型官方实现
- **类型：** repo
- **机构：** 谷歌研究院（Google Research）
- **代码：** <https://github.com/google-research/timesfm>
- **权重：** <https://huggingface.co/google/timesfm-3.0-pytorch>
- **发布：** TimesFM 3.0（2026-08）
- **入库日期：** 2026-11-21
- **核对范围：** 通读官方 README（版本历史、安装、PyTorch/MLX 预测示例、benchmark、许可声明），查看 pyproject.toml 依赖、src/timesfm3 模块布局与 torch/evaluator.py、torch/timesfm3_forecaster.py 的公开接口；未下载权重、未运行本地 GPU benchmark。
- **通读：** 已完成

## 已核验入口

| 入口 | 实际用途 |
|------|----------|
| README.md | 2.5 / 3.0 版本与授权声明、PyTorch/MLX 使用示例 |
| pyproject.toml | Python >= 3.10；包版本 3.0.2；可选 torch、mlx、flax |
| src/timesfm3/torch/timesfm3_forecaster.py | TimesFM3Forecaster / ModelConfig / ForecastOutput，预测与协变量接口 |
| src/timesfm3/torch/evaluator.py | TimesFM3Evaluator（基准评估默认参数，委托 forecaster） |
| src/timesfm3/torch/model.py | PyTorch 模型组件 |
| src/timesfm3/mlx/ | Apple Silicon MLX 原生推理后端 |
| huggingface.co/google/timesfm-3.0-pytorch | 官方 3.0 checkpoint（使用前需核对权重授权） |

## 安装与推理接口（README）

- 安装 PyTorch 后端：`pip install 'timesfm[torch]'`；Apple Silicon 可用 `pip install 'timesfm[mlx]'`。安装包不代表已经下载了权重。
- 3.0 README 示例：`from timesfm3 import TimesFM3Evaluator, ModelConfig`，`ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device="cuda", per_core_batch_size=32)`；`predict_batch([...], horizon=12, return_quantiles=True)`。
- 单变量输入为 NumPy 1D 数组；多变量输入为 `(num_variates, context_length)`，可搭配 `past_only_covariates` 与 `past_future_covariates`，后者需包含未来已知段。
- 返回 `ForecastOutput.forecast` 与可选 `quantiles`；官方示例的分位数 shape 为 `(horizon, 9)`。
- README 报告 M4 Max、context=512、horizon=64 的 MLX 性能为 batch 1 约 11.1ms，**仅该机器/配置的官方测量**，不应泛称所有本地设备均轻量毫秒级。

## 授权特别说明（重要）

| 范围 | 官方许可 |
|------|----------|
| 仓库代码 | Apache-2.0 |
| 截止 2.5 的模型权重 | Apache-2.0 |
| 3.0 下载／自托管权重 | `timesfm-non-commercial-license-v1.0`；**仅非商业、非生产** |
| 3.0 商业生产使用 | 通过 Google 认可的云服务（如 BigQuery ML），遵循相应云服务条款 |

> “开源代码可免费获取”不等于“3.0 预训练权重可以免费商用”。

## 对 wiki 的映射

- [TimesFM 3.0 项目实体](../../wiki/entities/timesfm-3.md)
- [Google Research 官方解析](../blogs/google-research-timesfm3-2026.md)
