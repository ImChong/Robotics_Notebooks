# 机器人学习数据管线：一手资料核对

- **通读：** 已完成
- **核对范围：** 2026-11-26，Google Research RLDS README 的数据格式、采集与载入部分，Open X-Embodiment 官方项目页的数据集和模型小节，LeRobotDataset 3.0 官方文档的文件布局、episode 索引与载入窗口。这里只复核接口及边界，未下载完整数据或运行训练。

## 一手来源

- [Google Research RLDS：Dataset Format 与 EnvLogger 时间对齐](https://github.com/google-research/rlds#dataset-format)：episode 由有序 step 组成；最后一个 step 的 action/reward/discount 不应当作为该观测的监督目标；终止和截断是不同语义。EnvLogger 生成的记录与 RLDS 读出格式对动作、奖励的归属存在位置差异，转换时不能直接按行拷贝。
- [Open X-Embodiment 官方项目](https://robotics-transformer-x.github.io/)：统一格式汇总多本体演示；作者展示的 RT-X 输出对齐到末端夹爪坐标系的七维向量，未使用的维度填零。统一容器本身不意味着各源动作语义、传感器和许可一致。
- [Hugging Face LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)：低维数据使用 Parquet、图像主要使用视频文件；episode 通过元数据定位所跨的文件与时间范围，不再简单等同于一个文件。时间窗口由 loader 按声明的相对时间偏移读取。
