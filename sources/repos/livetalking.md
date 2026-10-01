# LiveTalking（GitHub 主仓）

> 来源归档（ingest）

- **标题：** LiveTalking — Real time interactive streaming digital human
- **类型：** repo
- **仓库：** <https://github.com/lipku/LiveTalking>
- **项目页：** <https://www.livetalking.ai>
- **文档：** <https://doc.livetalking.ai>
- **许可：** Apache-2.0（README badge，2026-10-01）
- **入库日期：** 2026-10-01
- **一句话说明：** **实时交互流式数字人引擎**：文本/语音 → 可选 LLM → TTS → 口型推理（Wav2Lip / MuseTalk / ER-NeRF 等）→ WebRTC / RTMP / 虚拟摄像头输出；支持打断、多并发与 HTTP API 对接。

## 开源状态（项目页 + GitHub 核查，2026-10-01）

| 项 | 状态 |
|----|------|
| 推理与服务代码 | **已开源** — [lipku/LiveTalking](https://github.com/lipku/LiveTalking)，Apache-2.0 |
| 预训练权重 / Avatar 资产 | **部分** — README 提供夸克 / Google Drive 网盘，需自行下载至 `models/`、`data/avatars/` |
| 商业版增强能力 | **未开源** — 项目页「商业版」列 Wav2Lip 升级版、全语音交互、透明背景等，源码在付费/定制线 |
| Docker 镜像 | **已提供** — AutoDL / UCloud 等第三方 GPU 镜像（非 GitHub Releases 一体包） |

## 核心能力（README 编译）

- 多数字人后端：ernerf、musetalk、wav2lip、Ultralight-Digital-Human 等
- 声音克隆；说话可打断；全身视频拼接；不说话时 idle 动作编排
- 输出：WebRTC、RTMP、虚拟摄像头；多 session 并发
- 插件：`registry.py` 注册 TTS / Avatar / Output 模块
- 关联仓：[livestream](https://github.com/lipku/livestream)（直播带货场景）

## 典型复现路径

```
conda 环境 + PyTorch(CUDA) → pip install -r requirements.txt
  → 网盘下载 wav2lip.pth 与 avatar 包 → data/avatars/
  → python app.py --transport webrtc --model wav2lip --avatar_id <id>
  → 浏览器 index.html 或 POST /human、/offer（见 docs/api.md）
```

**端口：** 服务端 TCP 8010；WebRTC 需开放 UDP 大范围（README 注记）。

## 对 wiki 的映射

- [wiki/entities/livetalking.md](../../wiki/entities/livetalking.md)
- [sources/sites/livetalking-ai.md](../sites/livetalking-ai.md)
