# 运动控制主路线讲解视频：源稿与生成脚本

把 [主路线：运动控制 → 物理智能全栈成长路线](../../roadmap/motion-control.md)（站点 `roadmap.html?id=roadmap-motion-control`）做成一支逐级讲解视频：L−1 → L0–L7 → L8–L12，每一层讲清核心要点与原理，路线里的自测题直接在讲解中给出答案。

- 约 100 分钟、24 章（MP4 内嵌章节标记），1920×1080，中文配音 + 烧录字幕，另出 SRT 字幕与章节时间表。
- 成片入库于 [`docs/assets/video/`](../../docs/assets/video/)（约 83 MiB，低于 GitHub 单文件 100 MB 上限），嵌在主路线页首可直接播放；用本目录可完整复现。

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
2| 每读{一行|一航}都要查    ← {字幕写法|念法}：字幕显示「一行」，配音念「一航」
```

## 读音与字幕（改稿时注意）

做法参考 [Robot_Learning_Paper_Notebooks 的论文视频](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/tree/main/scripts/paper_video)：拿不准的字，用同一整句分别合成原字和各读音的同音字，按词边界截出那个词，比 log-mel DTW，离谁近就是念成了谁。

- 整句里会读错的多音字，用 `{原字|同音字}` 只改配音：「每一行」会念成 xíng（写 `{一行|一航}`），「重规划」「概率调高」念成 zhòng / diào（写 `{重|虫}`、`{调|条}`），「当成」念成 dāng（写 `{当|荡}`）。DTW 分不出来的（如「重定向」）也直接写同音字。
- 能拼成英文单词的缩写会被当单词念（LIP 念成 lip、GAE、PoE、ABA、CoP、AMP 等），在 `build.py` 的 `PRON` 里改成逐字母；EtherCAT 写成 Ethercat，免得 CAT 被逐字母念。
- 字幕不在英文词、数字中间断行，也尽量不把「Isaac Sim」「ONNX Runtime」这类多词英文名拆到两行。
- 编码按 BT.709 矩阵转换并写明色彩标记（未标注时多数播放器按 BT.709 解释高清视频，按 BT.601 编出来的青色会偏）；时长按总帧数截断，音画等长。

## 复现

```bash
cd media/roadmap-motion-control-video
npm install
pip install edge-tts imageio-ffmpeg numpy jieba    # jieba 可选：字幕按词断行
CHROME_PATH=/path/to/chrome python3 build.py all   # 产物在 out/
```

- 配音用 [edge-tts](https://github.com/rany2/edge-tts)（微软在线 TTS，需联网），默认 `VOICE=zh-CN-YunxiNeural`、`RATE=+6%`；经 TLS 代理访问时设 `TTS_CA_BUNDLE=<证书链路径>`。
- 配音与帧按内容哈希缓存在 `cache/`，改稿后只重做变动部分；`python3 build.py all c08` 只出某一章。
- 改稿后先出片段给人看：`python3 build.py clip c08-042,c08-044:1+2 clip-c08` 只出这几张幻灯片（`:1+2` 表示只要该页第 1、2 句旁白），每张一个章节标记。
- 路线原文更新后，改对应的 `chapters/*.txt` 再重跑即可。
