# Source: G1 Football（NVlabs/ProtoMotions 浏览器演示发布与安装）

- **类型：** 官方仓库的独立 demo 子模块（compiled application + 可运行部署/验证脚本）
- **代码：** https://github.com/NVlabs/ProtoMotions
- **模块：** https://github.com/NVlabs/ProtoMotions/tree/main/demos/g1-football
- **官方 README：** https://github.com/NVlabs/ProtoMotions/blob/main/demos/g1-football/README.md
- **安装脚本：** https://github.com/NVlabs/ProtoMotions/blob/main/scripts/install_g1_football.py
- **浏览器预览验证：** https://github.com/NVlabs/ProtoMotions/blob/main/scripts/check_g1_football_browser.py
- **官方文档：** https://nvlabs.github.io/ProtoMotions/user_guide/g1_football.html
- **演示地址：** https://nvlabs.github.io/ProtoMotions/demos/g1-football/
- **基础模型：** https://huggingface.co/nvidia/Metatron-v1
- **机构：** 英伟达（NVIDIA / NVlabs）
- **核对日期：** 2026-11-12（仓库 main；未指定发布标签）
- **核对范围：** 通读官方 demos/g1-football/README.md 全文（Release status、Controls、Local fixture preview、Known limits and licenses）；检查 scripts/install_g1_football.py 的 metadata、verify、check_policy、check_perception、check_dependencies、check_notices、install 与 CLI main；检查 scripts/check_g1_football_browser.py 的入口、HTTP/Playwright 测试及权重预加载拦截逻辑。未读取压缩资源包内部编译 JS/Python 字节码，未运行浏览器游戏。
- **通读：** 已完成

## 已发布与未发布的边界

| 内容 | 公开状态 | 核查说明 |
|---|---|---|
| ProtoMotions 通用框架 | 已公开源码 | Python 框架主仓库，不能直接当作 G1 Football 全部游戏源码 |
| G1 Football 浏览器游戏 | 已公开**编译包** | `demos/g1-football/game.zip`（Git LFS）及包清单 / SHA256 / NOTICE |
| 球感知实现 | **仅编译字节码** | `keeper-perception-cp313.zip`；与 Pyodide 3.13 运行时配套，源码和映射不在发布包中 |
| G1 Football 交互前端 | **仅编译 JS 等资源** | 资源包可查看，不含开发源码或 source map |
| Metatron 模型 | 公开权重 | 按不可变 Hugging Face revision 下载约 403 MB，模型权重受 NVIDIA Open Model License 约束 |
| 安装与静态预览脚本 | 公开 Python 源码 | `scripts/install_g1_football.py` 与 `scripts/check_g1_football_browser.py` 可逐文件分析 |
| 动作参考 | 发布包内衍生片段 | 远端扑救引用一条 BONES-SEED 动作，NOTICE 有归属说明，不等于公开整个训练集 |

## 官方本地安装入口

README 的本地预览命令（需获取含 LFS 资源的仓库）：

```bash
git lfs pull --include="demos/g1-football/game.zip"
python scripts/install_g1_football.py --allow-fixture --out /tmp/football-site/demos/g1-football
python -m http.server 4188 --bind 127.0.0.1 --directory /tmp/football-site
# 浏览 http://127.0.0.1:4188/demos/g1-football/
```

- `--allow-fixture` 兼容旧用法；已发布包标记 `release_status: public-release`，无该开关也可通过安装入口。
- 安装器先读取 `bundle.json`，核对 release status、文件清单、ZIP 安全路径/大小/哈希、嵌套 Pyodide ABI 与依赖锁、感知字节码、模型 manifest 固定权重 URL / SHA256、许可证及 attribution，最后仅解压编译资源，**不下载模型权重**。
- 服务静态目录后，由页面依次初始化场景、视觉运行时和 Metatron；Service Worker 配置跨源隔离；WebGPU 优先，CPU/WASM 备用（`?provider=wasm`）。
- 可用 `python scripts/install_g1_football.py --check` 检查捆绑包或 `--status` 检查公开发布标记。

## 官方预览验证入口及限制

```bash
python -m pip install playwright==1.55.0
python -m playwright install chromium
python scripts/check_g1_football_browser.py /tmp/football-site
```

该脚本用 `ThreadingHTTPServer` 提供静态文件，Playwright 打开 Chromium，检查 `/demos/g1-football/` 子路径、隔离、场景/相机、本地感知环境与背景预加载消息。它在 `Worker.postMessage({type: "start"})` 边界截断真正的权重初始化，因此 **PASS 不是策略可用、扑救成功率或真实移动设备兼容性的证据**。

## 开发者可复核的最短链路

1. 查发布状态：`python scripts/install_g1_football.py --status`。
2. LFS 获取游戏 ZIP 后用安装器核对并解压；许可证与依赖清单须随包保留。
3. HTTP 本地部署，测试浏览器的 Worker/Service Worker 初始化及输入门控。
4. 完整实际游戏需加载官方模型权重并执行交互；公开预览测试刻意不做这一步。编译游戏内部策略调用没有同等可读源码可追踪。

## 对知识库的映射

- [G1 Football 独立演示实体](../../wiki/entities/g1-football-metatron-goalkeeper.md)
- [官方演示技术文档归档](../sites/g1-football-protomotions.md)
- [ProtoMotions 框架实体](../../wiki/entities/protomotions.md)
