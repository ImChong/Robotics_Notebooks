# 运动控制主路线讲解视频：源稿与生成脚本

把 [主路线：运动控制 → 物理智能全栈成长路线](../../roadmap/motion-control.md)（站点 `roadmap.html?id=roadmap-motion-control`）做成一支逐级讲解视频：L−1 → L0–L7 → L8–L12，每一层讲清核心要点与原理，路线里的自测题直接在讲解中给出答案。

- 约 96 分钟、24 章（MP4 内嵌章节标记），1920×1080，中文配音 + 烧录字幕，另出 SRT 字幕与章节时间表。
- 成片入库于 [`docs/assets/video/`](../../docs/assets/video/)（约 85 MiB，低于 GitHub 单文件 100 MB 上限），嵌在主路线页首可直接播放；用本目录可完整复现。

## 目录

| 路径 | 内容 |
|------|------|
| `chapters/*.txt` | 讲解稿：每张幻灯片的 HTML + 分步旁白，页脚注明依据的路线章节与 wiki 页 |
| `svg/*.svg` | 原理示意图（倒立摆、支撑多边形、DCM、质心动量、PPO clip、流匹配等） |
| `template.html` / `style.css` | 1920×1080 幻灯片模板（KaTeX 公式、顶部 L−1…L12 进度条、字幕条） |
| `render.cjs` | 用 Chromium 把每一步 + 每句字幕截成 PNG 帧 |
| `build.py` | 解析讲解稿 → 配音 → 渲染 → ffmpeg 合成 MP4 / SRT / 章节表 / 旁白稿 |

## 讲解稿格式

```text
# chapter: id=c08; level=L4; tag=L4.1; name=L4.1 LIP / ZMP：会走路的倒立摆
## slide: title=线性倒立摆：三个假设，一个线性方程; src=roadmap/motion-control.md §L4.1 · wiki/…
<HTML：data-s='k' 的元素在第 k 步出现；::svg 文件名 内联 svg/ 下的图；::chap … 生成章节封面>
---
0| 第 0 步时念的一句旁白（同时是字幕）
1| 第 1 步时念的旁白……
```

## 复现

```bash
cd media/roadmap-motion-control-video
npm install
pip install edge-tts imageio-ffmpeg numpy jieba    # jieba 可选：字幕按词断行
CHROME_PATH=/path/to/chrome python3 build.py all   # 产物在 out/
```

- 配音用 [edge-tts](https://github.com/rany2/edge-tts)（微软在线 TTS，需联网），默认 `VOICE=zh-CN-YunxiNeural`、`RATE=+6%`；经 TLS 代理访问时设 `TTS_CA_BUNDLE=<证书链路径>`。
- 配音与帧按内容哈希缓存在 `cache/`，改稿后只重做变动部分；`python3 build.py all c08` 只出某一章。
- 路线原文更新后，改对应的 `chapters/*.txt` 再重跑即可。
