# LingBot-Touch — 官方代码、权重与推理入口核验

- **代码：** <https://github.com/Robbyant/lingbot-touch>
- **README：** <https://github.com/Robbyant/lingbot-touch/blob/main/README.md>
- **输入指南：** <https://github.com/Robbyant/lingbot-touch/blob/main/DATA_INTERFACE_GUIDE.md>
- **入口：** <https://github.com/Robbyant/lingbot-touch/blob/main/infer.py>
- **调用：** <https://github.com/Robbyant/lingbot-touch/blob/main/data_interfaces/infer.py>
- **encoder：** <https://github.com/Robbyant/lingbot-touch/blob/main/tactile_jepa_inference/model.py>
- **Tiny：** <https://huggingface.co/robbyant/lingbot-touch-tiny>
- **Small：** <https://huggingface.co/robbyant/lingbot-touch-small>
- **ModelScope：** <https://www.modelscope.cn/collections/Robbyant/LingBot-Touch>
- **核验日期：** 2026-11-10
- **核对范围：** README 全文、DATA_INTERFACE_GUIDE.md 的安装/输入契约/Sharpa/XHand/其他传感器段落、infer.py、data_interfaces/infer.py、tactile_jepa_inference/model.py、pyproject.toml、HF 两份模型卡及其权重/配置清单；用于推理流程的源码已阅读，非整仓代码逐行审计。
- **通读：** 已完成
- **许可：** GitHub `LICENSE` 为 Apache-2.0；第三方保留各自许可，详情见 THIRD_PARTY_NOTICES.md。
- **开放程度：** **部分开源（推理运行路径 + Tiny/Small 冻结 target encoder 权重）**；README 称 inference release。HF 模型卡明确不包含 predictor、downstream task head、optimizer、训练日志/配置或训练数据；不能据此声称已开放完整预训练复现链。

## 实际代码调用链（可对应时序图）

1. `python infer.py --input-format sharpa|xhand|normalized --bundle ... --input ... --output ... --device cpu` → 顶层 `infer.py` 导入并调用 `data_interfaces.infer.main()`。
2. `main()` 检查 CLI 合法性及输出不可覆盖，使用 `torch.load(..., weights_only=True)` 加载输入；按类型选择 `prepare_sharpa`、`prepare_xhand` 或 `prepare_normalized`。
3. Sharpa：标量触觉（不是 RGB）→ 固定 Z/255 归一化，已 canonical 的几何**不重复变换**；XHand：根据 handedness/`layout_id` 映射 120→96 点并对 XYZ 按 [128,128,255] 归一化。
4. `load_encoder(bundle, device=...)` 需要同一 bundle 的 `config.json` 与 `encoder.pt`；目标 encoder 冻结且 eval 模式。
5. `encode_tactile(encoder, signal, positions, modality, coverage_mask)` → `TargetEncoder.encode` → `ModalityRoutedTokenizer` → 可选 `SharedPointwiseProjector` → `AddressTransformer`。Transformer 注意力对 time/finger/XYZ 使用地址 RoPE，输出多个 layer features 的拼接。
6. `torch.save` 保存 `latent`、`time_id`、`finger_id`、`pos_xyz`；不在此调用任何 VLA 策略或机器人实时控制器。
7. 项目推荐 downstream 用 cross-attention / query 获取需要的触觉信息，**不要把 token 平均池化默认当作指尖物理量**。

## 输入输出契约

| 字段 | 核查结果 |
|---|---|
| signal | 浮点 `[B,T,5,M,3]`，单样本支持 `[T,5,M,3]` |
| pos_xyz | `[5,M,3]` 或 `[B,5,M,3]`，必须对应传感器物理布局的 canonical 坐标 |
| modalities | `magnetic` / `piezoresistive` / `visuotactile` |
| 数值 | XY 在 [-1,1]；Z 在 [0,1]，标量信号使用 [0,0,Z] |
| coverage_mask | `[5,M]` 的真实 taxel 存在掩码，**不是接触 mask** |
| 手指 | thumb,index,middle,ring,pinky；左右手各自规范化，不能叠成 10 指输入 |
| 输出 | `latent [B,N,C]` + 时/指/空间地址；具体 C 依 bundle 配置而定 |

## 官方最小复现命令（推理演示，非论文训练复现）

```bash
conda create -n lingbot-touch python=3.12.13 pip=26.0.1 -y
conda activate lingbot-touch
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
python -m pip install .
# 另行下载匹配的 config.json 和 encoder.pt 到 checkpoints/lingbot-touch-tiny/
python infer.py --input-format sharpa --bundle checkpoints/lingbot-touch-tiny --input sample_data/sharpa/001_left.pt --output outputs_left.pt --device cpu
```

官网 README 标注 Linux / Python 3.12 / PyTorch 2.11 + CUDA 13.0 验证环境；本次是源码级静态核验，**没有本机下载权重或执行 GPU/CPU 推理**。

## 对 Wiki 的映射

- [LingBot-Touch 主实体](../../wiki/entities/paper-lingbot-touch.md)
- [项目页核验](../sites/lingbot-touch-official.md)
- [技术报告阅读状态](../papers/lingbot-touch-technical-report.md)
