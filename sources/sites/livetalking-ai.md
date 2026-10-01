# LiveTalking.ai 项目页

- **类型**：网站 / 产品主页（开源社区版 + 商业版）
- **入口**：<https://www.livetalking.ai>
- **文档**：<https://doc.livetalking.ai>
- **代码：** <https://github.com/lipku/LiveTalking>（页眉/定价区链至 `lipku/livetalking`，与主仓同项目）
- **收录日期**：2026-10-01
- **抓取说明**：以 **2026-10-01** 对首页 HTML（Next.js 静态文案）与 README 交叉为准。

## 一句话

**LiveTalking** 定位为 **AIGC 时代实时交互数字人基础设施**：全链路流式架构、宣称 **亚 500ms 级延迟**、多模型融合（Wav2Lip / MuseTalk / ER-NeRF / Dinet 等），社区版 **Apache-2.0 开源**，商业版提供私有化与增强模型。

## 开源核查（步骤 2.5）

| 检查点 | 结论 |
|--------|------|
| 首页 GitHub 入口 | **有** — 指向 `https://github.com/lipku/livetalking` |
| 社区版定价卡片 | **核心源码开放**、标准 Wav2Lip/MuseTalk/ER-NeRF、WebRTC/RTMP |
| 商业版 | **闭源增强** — 升级版 Wav2Lip、唤醒词全语音、透明背景、多路独立配置等 |
| 权重与形象 | 项目页未托管模型文件；README **网盘外链**（部分开源边界） |

## 公开产品叙事（编译）

### 核心技术（首页 #features）

- **全双工实时交互**：可打断；WebRTC（P2P/SRS）与虚拟摄像头
- **多模型融合**：Wav2Lip（高并发）、MuseTalk（高精度嘴型）、Dinet、Ultralight / ER-NeRF
- **企业扩展**：GPT-SoVITS 等声音克隆、全身拼接、Idle Video 动作编排

### 落地场景（#solutions 摘要）

- 在线教育 / 文化讲解（RiverEcho + LLM/RAG）
- 7×24 直播带货
- 政务 / 金融私有化客服

### 生态入口

- API 文档 → `doc.livetalking.ai`
- Docker → UCloud / AutoDL 镜像链接（README 同步）
- 商用试用 → `https://www.livetalking.top`

## 对 wiki 的映射

- [wiki/entities/livetalking.md](../../wiki/entities/livetalking.md)
- [sources/repos/livetalking.md](../repos/livetalking.md)
