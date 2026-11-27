# TIPO 官方实现

- **类型：** repo
- **原文：** https://github.com/davidegardenal/TIPO
- **固定提交：** [15494a9c2b6d1fd5f9d6d0644896bbcfa67abbce](https://github.com/davidegardenal/TIPO/tree/15494a9c2b6d1fd5f9d6d0644896bbcfa67abbce)
- **许可证：** 仓库根 CC BY 4.0；SAM 子模块、模型权重和服务各自受其条款约束。
- **项目页：** https://davidegardenal.github.io/tipo-tracking/
- **论文：** https://arxiv.org/abs/2611.27398v1
- **数据集：** https://huggingface.co/datasets/davidegarde/VISTA
- **核对范围：** README 安装、数据和评测；追踪 main.py、SAM 3 配置、tracker/tipo.py、Judge、Labeler、StateGraph 和 eval.py 对应路径。
- **通读：** 已完成
- **核对日期：** 2026-11-27

- `main.py` 读取数据划分与帧率元数据。VISTA 的对象提示由 `metadata.json` 驱动，允许在后续帧注入；VOST 的独立跟踪入口使用第 0 帧对象。SAM 3 checkpoint 需要在 Hugging Face 获取访问许可。
- `track_transformations` 每帧分割、抽样和打分。每个对象先判断拆卸；装配选择须避开已被拆卸的 ID。接受的事件替换跟踪对象并写入结构图。公开代码的拆卸组内评分是逐分量对自身和组内其他像素的覆盖率相乘并取均值，不是论文式 (6) 的整体并集面积覆盖；不要把论文公式和发布代码的分数认作逐项相同。
- Judge 用当前标注帧及约 5/3/1 秒前的历史帧，发布版 Judge 把历史帧作为本机路径交给服务。Labeler 在完整结构跟踪之后可选地补名字和动作，也能产出无语义 JSON 再单独补标。默认论文使用 GPT-5.4；README 示例用 Qwen 3.5 35B 做本地 Labeler，两者的语义指标不能混用。
- 固定源码中提取 `get_compatible_assembly` 与 `safe_lin_sum_assignment`，用 NumPy/SciPy 在 CPU 完成 4 条合成断言；另核查公开元数据与摩卡壶标注 3 条。未运行 SAM、MLLM、GPU 跟踪或全量 VISTA 指标。
