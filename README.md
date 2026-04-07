 # Privacy-inference（AloePri 复现仓库）
 
 本仓库以 [Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf](file:///home/shengfeng/Privacy-inference/docs/Towards%20Privacy-Preserving%20LLM%20Inference%20via%20Collaborative%20Obfuscation%20(Technical%20Report).pdf) 为目标，逐阶段复现 AloePri 的协变混淆（Covariant Obfuscation）推理链路，并提供可运行的脚本、产物导出与回归测试。
 
 ## 文档入口
 
 - 接手/开发必读技术文档：[仓库技术文档.md](file:///home/shengfeng/Privacy-inference/docs/%E4%BB%93%E5%BA%93%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3.md)
 - 复现进度与实验报告（阶段 A-H）：[完整复现总报告_阶段A-H.md](file:///home/shengfeng/Privacy-inference/docs/%E5%AE%8C%E6%95%B4%E5%A4%8D%E7%8E%B0%E6%80%BB%E6%8A%A5%E5%91%8A_%E9%98%B6%E6%AE%B5A-H.md)
 - 部署说明（Stage H 产物）：[阶段H_混淆模型部署说明.md](file:///home/shengfeng/Privacy-inference/docs/%E9%98%B6%E6%AE%B5H_%E6%B7%B7%E6%B7%86%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E8%AF%B4%E6%98%8E.md)
 
 ## 快速开始
 
 - 环境文件：[environment.qwen-transformers.yml](file:///home/shengfeng/Privacy-inference/environment.qwen-transformers.yml)
 - 默认模型目录：`model/Qwen2.5-0.5B-Instruct`
 
 常用命令（按仓库已有脚本约定）：
 
 ```bash
 conda run --no-capture-output -n qwen-transformers pytest -q
 conda run --no-capture-output -n qwen-transformers python scripts/run_stage_h_joint_regression.py --layer-count 24
 ```
 
 ## 目录结构速览
 
 - `src/`：核心实现（KeyMat、Attention、FFN、Norm、Stage 逻辑、评估工具）
 - `src/aloepri/`：新抽象的模块化引擎（将分阶段逻辑抽象为可复用组件）
 - `scripts/`：可运行入口（回归、导出、推理）
 - `tests/`：pytest 回归用例
 - `artifacts/`：导出的服务端/客户端产物（用于部署与离线推理）
 - `outputs/`：实验与回归输出 JSON
