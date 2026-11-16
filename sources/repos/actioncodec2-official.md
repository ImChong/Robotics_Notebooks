# ActionCodec2 官方源码与公开运行链路

- **类型：** 代码仓库
- **仓库：** https://github.com/ZibinDong/actioncodec2
- **论文：** https://arxiv.org/abs/2611.15490
- **项目：** https://zibindong.github.io/actioncodec2-page/
- **开放状态：** 公开可运行 tokenizer 与 VLA 训练/评测相关模块；主仓 Apache-2.0，G0.5 派生代码/策略受 LICENSE-G0.5 非商用限制。
- **核对范围：** README；`scripts/{serve.py,benchmark_client.py,train.py}`；`src/actioncodec2/vla/runtime/{server.py,policy.py}` 及 `src/actioncodec2/tokenizers/actioncodec2.py` 模块。
- **通读：** 已完成
- **wiki：** [ActionCodec2](../../wiki/entities/paper-actioncodec2-streaming-action-tokenizer.md)
- **原文：** [论文归档](../papers/actioncodec2-arxiv-2611-15490.md)
- **官方站点：** [项目归档](../sites/actioncodec2-project.md)

## 入口与交互（静态核读，不代表执行复现）

1. 安装 Python 3.12 + C++17，README 指令 `uv pip install -e '.[cuda]'`。
2. `python scripts/serve.py --model ZibinDong/G05_ActionCodec2-2nd_LIBERO --trust-remote-code` → `actioncodec2.vla.runtime.server.main`，建立 WebSocket 策略服务。
3. `python scripts/benchmark_client.py --benchmark libero --benchmark-path /path/to/LIBERO --suite libero_spatial --episodes 50` → `PolicyClient` 请求 `metadata`/`predict`，用 `execution_horizon` 按步释放服务端返回的动作数组。
4. `ActionPolicy` 调用模型 `generate_action_tokens` 并使用 tokenizer `decode_batch` 转换到原始动作。
5. `python scripts/train.py --config configs/train/libero/actioncodec2.yaml --data-root /path/to/datasets` → `actioncodec2.vla.train.main`；需要外部数据和对应模型访问权限。

这里描述的是公开 LIBERO benchmark 路径，不等同于论文闭环真机实验脚本。
