# 《从头学人形机器人算法 · 零基础篇》视频：源稿与生成脚本

按 [发布清单](RELEASE_PLAN.md) 从 L−1 讲到 L5.1 的 PPO，一集只讲一个主题，竖屏 1080×1920，中文配音 + 烧录字幕，发视频号。讲完 060 接上姊妹系列《从头学人形机器人算法》（论文精读，[Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)）的 001 PPO。

渲染流程改自论文系列的 [`scripts/paper_video`](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/tree/main/scripts/paper_video)（BSD-3，见 `vendor/LICENSE`）：安全区、逐帧版式检查、旁白与分镜对时的做法都照搬。

## 和论文系列的区别

| | 论文系列 | 零基础篇 |
|---|---|---|
| 配色 | 灰黑底 `#16161a` + 蓝 `#569cd6` | 同一套灰黑底和蓝；系列色换成绿 `#4ec9a4`（角标「零基础篇」、幕号、要点框左边条） |
| 角标 | 「论文精读 · 机器人学习」+ arXiv 胶囊 | 「零基础篇 · 机器人学习」+ 季数胶囊（如「第一季」） |
| 封面 | 英文缩写大标题 + 论文信息框 + 目录 | 同版式：中文大标题 + 副标题 + 一句话 + 信息框 + 目录；不写集数和路线层级，只写第几季 |
| 片头 / 片尾 | 论文卡 + 目录 / 「三句话记住」 | 问句 + 主图 / 自测题（倒数后揭晓答案）+ 下一集预告 |
| 要点框 | 「要点」 | 「划重点」 |
| 配音 | `zh-CN-YunxiNeural` +22% | `zh-CN-YunjianNeural` +6% |

## 目录

| 路径 | 内容 |
|------|------|
| `episodes/<ep>/*.js` | 一集的分镜，按文件名顺序加载：共用部件 → 片头主图 → 各幕 → 最后一个文件组装 `window.EpisodeVideo`（集数、季、角标、问句、一句话、主图 `hero`、各幕 `scenes`、片尾 `outro`）。短的一集也可以只写一个 `episodes/<ep>.js` |
| `episodes/<ep>.py` | 旁白：`SCRIPT`（每句绑定一幕的分镜时间 `[from, to]`）与 `DISPLAY`（字幕换写法） |
| `stage.html` | 1080×1920 舞台：片头、分镜、片尾、封面版式，安全区 330–1440 |
| `build.py` | 旁白 → edge-tts 逐句配音 → `out/<ep>/timeline.json` + `narration.wav`。每句记下配音实长，舞台按它把这句的分镜区间等比拉伸，画面事件跟着念到的词走 |
| `render.mjs` | Chromium 逐帧截图 → ffmpeg 合成 MP4；也出抽帧、片段、封面，`scene` 模式不用配音单看某一幕 |
| `check_layout.mjs` | 逐帧检查：块不重叠、不压字幕、都在安全区；封面检查 3:4 置顶裁剪 |
| `vendor/` | 论文系列的分镜工具库 `kit.js` 与样式 `paper-demos.css`（取自 `f7a5b14`，未改动） |

## 用法

```bash
cd media/zero-to-ppo-video
npm install
pip3 install edge-tts imageio-ffmpeg
node render.mjs ep001 scene 3 0,5,12   # 做分镜时：不配音，单看第 3 幕这几个分镜时刻
python3 build.py ep001                 # 旁白 TTS -> out/ep001/timeline.json + narration.wav
node render.mjs ep001 stills 5,40,90   # 先抽几帧看排版
node check_layout.mjs ep001            # 输出 no overlap 才算过
node render.mjs ep001 video 30         # -> out/ep001/ep001_video.mp4
node render.mjs ep001 cover            # -> out/ep001/cover.png
node check_layout.mjs ep001 cover
```

- Chromium：设置 `CHROME=/path/to/chrome`，否则用 playwright-core 默认路径。
- 走 TLS 代理时设置 `SSL_CERT_FILE`，`build.py` 会让 edge-tts 用这份 CA。
- 成片、封面、配音都在 `out/`，不入库。
