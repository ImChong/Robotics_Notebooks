# SuperTuxKart（supertuxkart/stk-code 官方实现）

> 来源归档

- **标题：** SuperTuxKart — The code base of SuperTuxKart
- **类型：** repo / game-engine / racing
- **组织：** SuperTuxKart Team
- **代码：** <https://github.com/supertuxkart/stk-code>
- **License：** **GPL**（根目录 [`COPYING`](https://github.com/supertuxkart/stk-code/blob/master/COPYING)；GitHub API 显示 `Other`）
- **项目主页：** <https://supertuxkart.net/>
- **Releases：** <https://github.com/supertuxkart/stk-code/releases>（最新稳定 **1.5**，2025-10-20）
- **入库日期：** 2026-09-18
- **一句话说明：** 开源卡丁车竞速游戏 C++ 代码仓；须与 SVN `stk-assets` 并排；CMake 构建，支持 Linux/Windows/macOS/Android/Switch；README 强调 **趣味优先、非真实卡丁物理**。
- **沉淀到 wiki：** [`wiki/entities/supertuxkart.md`](../../wiki/entities/supertuxkart.md)

---

## 双仓结构（Source control）

| 仓库 | VCS | URL | 必需性 |
|------|-----|-----|--------|
| **stk-code** | Git | `git clone https://github.com/supertuxkart/stk-code.git` | **必需** |
| **stk-assets** | SVN | `svn co https://svn.code.sf.net/p/supertuxkart/code/stk-assets` | **必需（游玩/编译）** |
| **stk-media-repo** | SVN | `media/trunk` on SourceForge | 可选（艺术家源文件 ~3.2GB） |

两仓须位于**同一父目录**，使 `stk-code/` 与 `stk-assets/` 并列。

---

## README / INSTALL 归纳

### 硬件

- OpenGL **≥ 3.3** 或 OpenGL ES **≥ 3.0**；Android **≥ 5.0**。
- 集成/独显约 2010+；Android 约 2014+ 设备可跑。

### 依赖（Linux 示例）

`cmake`, `SDL2`, `OpenAL`, `libcurl`, `libenet`, Ogg/Vorbis, Freetype/Harfbuzz, libpng/jpeg/zlib, OpenSSL/mbedTLS, Bluetooth（可选）等 — 见 [`INSTALL.md`](https://github.com/supertuxkart/stk-code/blob/master/INSTALL.md) 各发行版包名。

### 构建（Linux 典型）

```bash
git clone https://github.com/supertuxkart/stk-code stk-code
svn co https://svn.code.sf.net/p/supertuxkart/code/stk-assets stk-assets
cd stk-code
mkdir cmake_build && cd cmake_build
cmake ..
make -j$(nproc)
./bin/supertuxkart
```

- Vulkan/Shaderc：非 Windows/macOS 需自编译 [Shaderc](https://github.com/google/shaderc)，或 `cmake -DNO_SHADERC=on ..`。
- Android：见仓内 [`ANDROID.md`](https://github.com/supertuxkart/stk-code/blob/master/android/ANDROID.md)。

### 坐标系（README 提醒）

- **STK：** X 右、Y 上、Z 前（代码里地面为 **XZ**）。
- **Blender：** X 右、Y 前、Z 上；导出工具自动变换。

---

## 目录导航（复现相关）

| 路径 | 作用 |
|------|------|
| `src/` | 游戏与引擎源码 |
| `data/` | 指向/嵌入 assets 的配置 |
| `cmake/` | 构建模块 |
| `android/` | Android 构建说明 |
| `.github/workflows/` | Linux/Apple/Windows/Switch CI |
| `INSTALL.md` | 全平台编译指南 |

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [赛车漂移 RL 开源景观](../../wiki/overview/racing-drift-rl-open-source-landscape.md) | 与 CARLA/f1tenth **科研训练栈** 正交；属 **街机卡丁车可玩/可读源码** 样本 |
| [drive-game](../../wiki/entities/drive-game.md) / [starter-kit-racing](../../wiki/entities/starter-kit-racing.md) | 同为开源赛车/卡丁体验，STK 为 **原生多平台 + 在线多人** |
| 交互式世界模型 | Awesome WM「Game Engines」分组的对照：**完整可玩 3D 游戏** 而非 RL Gym |
