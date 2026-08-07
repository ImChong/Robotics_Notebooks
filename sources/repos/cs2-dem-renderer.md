# cs2-dem-renderer（Reka 官方）

> 来源归档

- **标题：** cs2-dem-renderer
- **类型：** repo
- **链接：** <https://github.com/reka-ai/cs2-dem-renderer>
- **数据集：** <https://huggingface.co/datasets/RekaAI/CS2-10k>
- **新闻页：** <https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset>
- **Viewer：** <https://huggingface.co/spaces/RekaAI/CS2-10k-viewer>
- **机构：** 瑞卡人工智能（Reka AI）
- **许可：** MIT
- **入库日期：** 2026-08-07
- **一句话说明：** 将 CS2 `.dem` 转为每玩家每回合 `.mp4` + 同步 parquet 标注的开源管线；含服务器插件、Go 渲染器与浏览器 Viewer。

---

## 运行时要点

1. **Parse（两遍）** — 提取出生/死亡 tick 区间与逐帧按键。
2. **Render** — 生成 JSON 动作序列，经 Steam 启动 CS2 + 服务器插件回放。
3. **Encode** — CS2 movie 输出 TGA 流式管道至 ffmpeg（`hevc_vaapi`）写 `.mp4`。
4. **Parquet** — 与视频对齐的逐帧元数据。

### 依赖（README）

- Go 1.21+、CMake 3.14+、C++ 编译器
- ffmpeg + VAAPI（`hevc_vaapi`）
- 经 Steam 安装的 Counter-Strike 2
- 插件需对齐 CS2 版本（主分支示例：1.41.6.5；其它版本见分支）

### 用法摘要

```bash
# 构建插件 + 安装
cmake ./cs2-server-plugin/ -B plugin-build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin-build
./install-plugin.sh plugin-build/libserver.so

# 构建渲染器
cd dem-render && go build -o dem-render .

# 单 demo（Steam 需先运行）
dem-render [flags] <path-to-demo.dem>

# 批处理 worker
dem-render worker --input /path/to/demos --output /path/to/output
```

浏览器 Viewer：打开 `viewer/index.html`（需 File System Access API）；可滤地图、同步回放动作与鼠标曲线、XY 小地图。

## 对 wiki 的映射

- **wiki/entities/rekacs2-10k-dataset.md** — 数据集与管线时序图
- **sources/datasets/rekacs2-10k.md** / **sources/sites/rekacs2-10k.md**
