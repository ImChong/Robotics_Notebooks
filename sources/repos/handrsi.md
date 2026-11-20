# HandRSI 官方代码：GinKenneth/Hand_RSI

- **代码：** https://github.com/GinKenneth/Hand_RSI
- **论文：** https://arxiv.org/abs/2611.22164 （[论文归档](../papers/handrsi_arxiv_2611_22164.md)）
- **权重：** https://huggingface.co/HandRSI/HandRSI （[模型归档](../models/handrsi-hf-checkpoints.md)）
- **核查日期：** 2026-11-20
- **核对范围：** 官方 README 全文、`infer_video.py` 输入输出及 81 帧分段、`rsi/rounds/rsi_loop.py` 各 stage 与停止条件、`ace_eval/README.md` 的 NPZ 格式和指标定义。未执行训练或 inference。
- **通读：** 已完成

## 公开范围与源码入口

| 路径 | 核实的作用 |
|---|---|
| `infer_video.py` | 视频 + 相机内参 → ACE-Ego-Hand `GeoDitModel` / VAE → 可选 Harness → 双手 MANO 的 `.npz` |
| `rsi/harness/` | 模块链 `chain.json`，H5* / Lite / Medium / Heavy 分层 |
| `rsi/rounds/rsi_loop.py` | 调度 `capture,base,loopA,teacher,integrate,loopB,revalidate,gate,sealed,controls`，随后 Loop C；连续两轮未过门控或 STOP 文件可终止 |
| `rsi/loopA/`、`rsi/loopB/`、`rsi/loopC/` | agent 提案与验证、蒸馏训练、实验反思 |
| `ace_eval/` | 根据 ACE-Ego-Hand 论文协议重写的 10 指标评估器；包含单元测试 |

## 运行与核查边界

```bash
# 依官方 README 先准备 ACE-Ego-Hand + VideoX-Fun/Wan2.2-Fun-5B-Control、
# ACE 基础 ckpt、MANO 与可选 WiLoR；再安装本仓库依赖。
pip install -r requirements.txt
huggingface-cli download HandRSI/HandRSI --local-dir checkpoints/handrsi
python infer_video.py --video clip.mp4 --intrinsics fx,fy,cx,cy \
  --ckpt checkpoints/handrsi/theta2 --harness rsi/harness/trees/H5 --out results/clip
# 轻量部署配置：--harness rsi/harness/trees/Lite
# 全循环（需要 root 沙箱、Agent API、8 张 GPU，非普通推理命令）
cd rsi && python rounds/rsi_loop.py --gpus 0,1,2,3,4,5,6,7 --start 1
```

- **视频 I/O：** 输入 RGB 视频 + K-given 相机标定；81 帧为推理分段；输出 `.npz` 的 `exists`、`joints_cam` (F,2,21,3)、`joints_2d`、`verts_cam`、`transl`、`rot`、`hand_pose`、`betas` 等，均在相机系。
- **重要依赖：** ACE-Ego-Hand 仓库、其第三方模型及训练权重、MANO 单独授权、WiLoR 可选模块；缺少 WiLoR 时，带其模块的 Harness 测得结果不能直接复用论文数字。
- **许可证：** 主仓代码 MIT；权重 CC BY-NC 4.0；模型依赖、MANO 和数据集另有许可。代码可用不意味着训练权重可商用。

- **Wiki：** [HandRSI](../../wiki/entities/paper-handrsi-recursive-self-improvement-hand-pose.md)
