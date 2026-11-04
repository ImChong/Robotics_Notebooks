# Jenkins 官方文档与源码

- **官方文档：** <https://www.jenkins.io/doc/>
- **Pipeline：** <https://www.jenkins.io/doc/book/pipeline/>
- **Jenkinsfile：** <https://www.jenkins.io/doc/book/pipeline/jenkinsfile/>
- **语法与触发器：** <https://www.jenkins.io/doc/book/pipeline/syntax/>
- **部署示例：** <https://www.jenkins.io/doc/pipeline/tour/deployment/>
- **源码：** <https://github.com/jenkinsci/jenkins>
- **类型：** 官方文档 / 开源自动化服务器
- **核查日期：** 2026-11-04

Jenkins 是开源自动化服务器；Pipeline 插件通过 Jenkinsfile 表达构建、测试和交付流程。文档区分 Declarative 和 Scripted 两种语法；触发方式包括 cron、pollSCM 和集成提供的 webhook。持续交付可包含人工审批，不必自动部署生产环境。

## 阅读范围与待完成

已核对用户指定的官方入口及 Pipeline 概览、触发器和部署示例；官方文档范围很大，尚未完整通读 Jenkinsfile 与语法参考全文。源码已核查 WAR 启动入口，但尚未追踪 Pipeline 插件运行路径。本条目保留为草稿。
