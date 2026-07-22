# AI 室内风水顾问

[English](README.md)

一个 Gradio 应用：上传一张房间照片，根据场景不同，输出风水指导的布局方案（毛坯/未装修空间），
或是带标注/编辑图片的问题清单（已装修空间）。为 NVIDIA DGX Spark 黑客松打造 —— 原始需求见
`Claude.md`。

## 架构

- **感知（Perception）**：YOLO-World（开放词汇版 YOLO，基于 `ultralytics`）通过自由文本类别提示
  检测门、窗、镜子、横梁和家具 —— 标准 COCO 类别里没有门/窗/镜子，而在冲刺时间内微调自定义检测器
  也不现实。
- **几何计算**（`perception/geometry.py`、`perception/facts.py`）：纯确定性 Python 实现 ——
  IoU、中心点连线角度、亮度、色彩平衡 —— 对应 `rules/rule_base.json` 中的 10 条规则。任何 LLM
  都不会自己计算几何关系，只负责把 Python 已经判定好的 `rule_verdicts` 转述成人话。
- **推理（Reasoning）**：DeepSeek（`api_clients/deepseek_client.py`）把事实数据转述成初稿建议；
  Step-3.7-Flash 通过阶跃星辰云端 "Step Plan" API（`api_clients/step_client.py`，OpenAI 兼容）
  把初稿格式化为严格的 schema JSON（`schema.py`）。Step 同样看不到原始几何数据。
- **生成（Generation）**：Wan2.7-Image（阿里云百炼 DashScope，`api_clients/wan_client.py`）——
  场景 A 用文生图，场景 B 用图像编辑；若物体"物理搬移"类指令效果不佳（扩散模型在严格空间位移上并
  不可靠），会自动降级为色彩/材质调整类指令。
- **语音（Audio）**：SenseVoice（DashScope）用于麦克风输入，`edge-tts`（本地、免费）用于朗读建议。
- **编排（Orchestration）**（`orchestrator.py`）：一个线性脚本，不是智能体框架 —— UI 上的
  毛坯/已装修切换直接决定走哪条路径，而非由模型动态决策。

## 环境搭建

```bash
uv sync
cp .env.example .env   # 填入 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY 和 STEP_API_KEY
```

Step-3.7-Flash 通过阶跃星辰云端 "Step Plan" API（`https://api.stepfun.com/step_plan/v1`，
OpenAI 兼容）运行 —— 无需本地下载模型或 GPU 部署。本项目早期版本曾尝试在这台单卡 DGX Spark 上用
`llama.cpp` 本地部署，但即便是最小可用的量化版本（约 92GB）在冲刺时间内下载也太慢；云端 API 完全
绕开了这个问题。

### 运行

```bash
./one-click-start.sh   # 启动 Gradio 应用（所有推理/生成均通过云端 API）
```

## 冒烟测试

在依赖完整流水线之前，先独立测试每个环节：

```bash
uv run python scripts/smoke_test.py --mode perception fixtures/raw_room.jpg
uv run python scripts/smoke_test.py --mode step
uv run python scripts/smoke_test.py --mode deepseek
uv run python scripts/smoke_test.py --mode wan
uv run python scripts/smoke_test.py --mode audio
uv run python scripts/smoke_test.py --mode e2e-a fixtures/raw_room.jpg "master bedroom"
uv run python scripts/smoke_test.py --mode e2e-b fixtures/furnished_bedroom_0.jpg
```

## 已知局限

- **延迟**：两次串行 LLM 调用（DeepSeek 转述 + Step 格式化）加上一次 Wan 图像生成调用，端到端耗
  时可能超过需求文档中原定的 10 秒目标。`orchestrator.py` 会记录每个阶段的耗时，便于提前发现慢
  环节，而不是在演示现场才发现。
- **镜子朝向检测**（`mirror_facing_bed_or_door` 规则）用 2D 包围盒 IoU 作为"朝向"的代理指标 ——
  没有深度/姿态估计。可能漏掉"2D 不重叠但实际朝向对着"的情况，偶尔也会误报。
- **横梁检测**：基于水平暗色条带的启发式方法在真实照片上噪声较大（阴影、灯具、石膏线都可能被误
  判为水平暗带）——建议作为参考性、低置信度的输出对待。
- **拍摄假设**：门窗夹角与财位亮度的启发式算法假设照片是较为正面、能覆盖多面墙的取景；斜角/广
  角手机照片会降低准确度。
- **`SEND_IMAGE_TO_LLM`** 默认关闭 —— Python 计算出的 `rule_verdicts` 已经是权威依据，把原始
  照片发给 VLM 会增加视觉 token 的预填充开销，直接挤压延迟预算。
- **Wan 调用超时**：DashScope SDK 自带的 `timeout` 参数并不生效（实测遇到过约 300 秒的静默挂起）；
  `wan_client.py` 用线程池加了一个真正生效的客户端侧超时（45 秒），超时后走降级流程，避免请求被
  无限期卡住。
