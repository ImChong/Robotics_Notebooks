# 《从头学人形机器人算法 · 零基础篇》视频：源稿与生成脚本

按 [发布清单](RELEASE_PLAN.md) 从 L−1 讲到 L5.1 的 PPO，一集只讲一个主题，竖屏 1080×1920，中文配音 + 烧录字幕，发视频号。讲完 060 接上姊妹系列《从头学人形机器人算法》（论文精读，[Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)）的 001 PPO。

渲染流程改自论文系列的 [`scripts/paper_video`](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/tree/main/scripts/paper_video)（BSD-3，见 `vendor/LICENSE`）：安全区、逐帧版式检查、旁白与分镜对时的做法都照搬。

## 和论文系列的区别

| | 论文系列 | 零基础篇 |
|---|---|---|
| 配色 | 灰黑底 `#16161a` + 蓝 `#569cd6` | 同一套灰黑底；系列色是亮黄 `#ffd166`：封面 / 片头大标题、角标「零基础篇」色块、目录符号、幕号、「划重点」框 |
| 角标 | 「论文精读 · 机器人学习」+ arXiv 胶囊 | 「零基础篇 · 机器人学习」+ 季数胶囊（如「第一季」） |
| 封面 | 英文缩写大标题 + 论文信息框 + 目录 | 同版式：中文大标题 + 副标题 + 一句话 + 信息框 + 目录；右上角胶囊写第几季，顶栏正中用灰色小字写集数「No.001」（只在封面上、不做最显眼的标志），不写路线层级 |
| 片头 / 片尾 | 论文卡 + 目录 / 「三句话记住」 | 问句 + 主图 / 总结卡片（`recap`，仿「三句话记住」，张数按内容定）+ 下一集预告 + 完整内容网址卡片（`outro.link`）+ 衔接语（不出题；`outro.quiz` 可选，零基础篇不写） |
| 要点框 | 「要点」 | 「划重点」 |
| 配音 | `zh-CN-YunxiNeural` +22% | `zh-CN-YunjianNeural` +6% |

## 目录

| 路径 | 内容 |
|------|------|
| `episodes/<ep>/*.js` | 一集的分镜，按文件名顺序加载：共用部件 → 片头主图 → 各幕 → 最后一个文件组装 `window.EpisodeVideo`（集数、季、角标、问句、一句话、主图 `hero`、各幕 `scenes`、片尾总结 `recap`、片尾 `outro`）。短的一集也可以只写一个 `episodes/<ep>.js` |
| `episodes/<ep>.py` | 旁白：`SCRIPT`（每句绑定一幕的分镜时间 `[from, to]`）与 `DISPLAY`（字幕换写法） |
| `stage.html` | 1080×1920 舞台：片头、分镜、片尾、封面版式，安全区 330–1440 |
| `build.py` | 旁白 → edge-tts 逐句配音 → `out/<ep>/timeline.json` + `narration.wav`。每句记下配音实长，舞台按它把这句的分镜区间等比拉伸，画面事件跟着念到的词走 |
| `render.mjs` | Chromium 逐帧截图 → ffmpeg 合成 MP4；也出抽帧、片段、封面，`scene` 模式不用配音单看某一幕 |
| `check_layout.mjs` | 逐帧检查：块不重叠、不压字幕、都在安全区；分镜 / 片头主图里的字、公式、图形、机器人不超出画面被裁掉，字和字（含公式）不互相压住；封面检查 3:4 置顶裁剪 |
| `vendor/` | 论文系列的分镜工具库 `kit.js` 与样式 `paper-demos.css`（取自 `f7a5b14`，未改动） |
| `robots.json` | 片中机器人的 URDF 来源：取自 [Robot Description Gallery Online](https://github.com/ImChong/Robot_Description_Gallery_Online)（固定 commit）的 `data/robots.json`，每台只记上游仓库 + 固定 commit 的 jsDelivr 地址与许可证。现用 Unitree G1（人形）、UR5（机械臂）、Unitree Go2（四足）、iRobot Create 2（扫地机）、Toyota Prius（汽车，OSRF car_demo）、Clearpath Jackal（轮式底盘） |
| `fetch_robots.py` | 按 `robots.json` 把 URDF 与可视网格下载到 `assets/robots/<id>/`（不入库），路径解析同画廊；上游只发布 xacro 的（Create 2、Jackal）连同它 include 的文件一起下载（照画廊的规则：`$(optenv)` 取默认值、`xacro_stubs` 里的包不下载），舞台加载时用 xacro-parser 展开 |
| `robot3d.js` | 舞台里的 URDF 渲染（three.js + urdf-loader，ES module）：全页共用一个离屏 WebGL 画布、正交相机，世界坐标线性映射到分镜坐标，叠加标注按正运动学定位；加载后用 meshoptimizer 减面到约 15%。另有和手画版同接口的封装：侧视 / 正面 G1、UR5、Go2、小人形图标、扫地机 / 汽车 / 轮式底盘，分镜里改样式的部件变成给连杆上色 |
| `serve.mjs` | 本目录的极简静态 http 服务：ES module 和 fetch 在 `file://` 下不能用，`render.mjs` / `check_layout.mjs` 都经它打开舞台 |

## 用法

```bash
cd media/zero-to-ppo-video
npm install
pip3 install edge-tts imageio-ffmpeg
python3 fetch_robots.py                # 下载片中机器人的 URDF 与网格（约 60 MB）-> assets/robots/
node render.mjs ep001 scene 3 0,5,12   # 做分镜时：不配音，单看第 3 幕这几个分镜时刻
python3 build.py ep001                 # 旁白 TTS -> out/ep001/timeline.json + narration.wav
node render.mjs ep001 stills 5,40,90   # 先抽几帧看排版
node check_layout.mjs ep001            # 输出 no overlap 才算过
node render.mjs ep001 video 30         # -> out/ep001/ep001_video.mp4（默认 3 段并行，第 4 个参数可改段数，1 = 一口气渲染）
node render.mjs ep001 cover            # -> out/ep001/cover.png
node check_layout.mjs ep001 cover
```

- Chromium：设置 `CHROME=/path/to/chrome`，否则用 playwright-core 默认路径。
- 走 TLS 代理时设置 `SSL_CERT_FILE`，`build.py` 会让 edge-tts 用这份 CA。
- 2026-10 起旧合成地址 `speech.platform.bing.com` 回 403，PyPI 上的 edge-tts 7.2.8 合成不了；7.3.1（2026-10-23 实测）已能正常合成，直接 `pip3 install edge-tts` 即可。
- 分段并行只在换幕处切（一幕里有的动画记着前面几帧），各段各开一个页面、各编一段无声视频，最后拼接再配旁白。001 全片（9356 帧）在 4 核机器上实测：1 / 2 / 3 / 4 段 = 530 / 392 / 360 / 342 s，逐帧和一口气渲染的版本比 PSNR 都 ≥ 50 dB。核数多的机器可以多分几段，最多到幕数。
- 成片、封面、配音都在 `out/`，不入库。
- URDF 机器人在无 GPU 的机器上走 Chromium 自带的 SwiftShader 软件渲染：G1 减面后约 8 万个三角形，一次重画约 50 ms；姿态和上色没变的视图不重画，移动、倾斜、倒地用 svg 变换，也不重画。一集用哪些机器人写在 `EpisodeVideo.robots`，舞台加载完才开始渲染。
