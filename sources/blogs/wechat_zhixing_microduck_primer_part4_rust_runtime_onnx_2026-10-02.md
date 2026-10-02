# 具身智能入门④：不碰硬件，把云端训出的 Microduck ONNX 塞进 Rust 运行时

> 来源归档（blog / 微信公众号 · 智践行）

- **标题：** 具身智能入门④：不碰硬件，把云端训出的 Microduck ONNX 塞进 Rust 运行时
- **类型：** blog
- **作者：** 智践行
- **日期：** 2026-09-28（frontmatter）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzk2NDU0MzA3OA==&mid=2247492524&idx=1&sn=03dfcab89f09e598d946cb073c69ae5e
- **专辑：** [wechat_zhixing_microduck_primer_album](../raw/wechat_zhixing_microduck_primer_album_4688586645438726146.md)
- **入库日期：** 2026-10-02
- **抓取方式：** wechat-article-for-ai（Camoufox，`--no-images`）**成功**；全文另存 [raw](../raw/wechat_zhixing_microduck_primer_part4_rust_runtime_onnx_2026-10-02.md)
- **一句话说明：** 在 Mac/本机用 `duck-control` 的 `policy-rehearsal` 离线加载 ONNX（`Policy::load` 即校验 61→14），装 ONNX Runtime 动态库，mock 轨迹推理并与 Python `infer_policy` CSV 对齐到 1e-9 量级。

## 核心摘录

### 仿真闭合 ≠ 部署闭合

- Python `onnxruntime` 能跑 ≠ Rust `robotd` 50 Hz 环能加载；壳差异：**动态库 dlopen**、观测维、NaN 守卫。
- **Everything is validated at load, not at inference**（`policy.rs`）。

### 最小命令

```bash
git clone https://github.com/pollen-robotics/microduck && cd microduck
export ORT_DYLIB_PATH=…/libonnxruntime.dylib   # load-dynamic；版本 ≥ 1.23
cargo run --release -p duck-control --example policy-rehearsal -- output.onnx
cargo run --release -p duck-control --example policy-rehearsal -- output.onnx trace.json
```

### 对齐验证

- 云上 `infer_policy.py --save-csv obs.csv` → 还原第 100 步观测（**obs[34:48] 用上一步 action**）→ `trace.json` → Rust 输出与 CSV action 最大差 ~4.9e-9。

### 故障排查

- `duck-control/tests/fixtures/bad_*.onnx` 共 10 个故意坏图，加载阶段即失败。

## 对 wiki 的映射

- 详情页：[zhixing-microduck-primer-part4-rust-runtime-onnx.md](../../wiki/overview/zhixing-microduck-primer-part4-rust-runtime-onnx.md)
- 实体：[pollen-microduck.md](../../wiki/entities/pollen-microduck.md)
