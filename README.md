# 自动光学镜头设计 Agent

一个基于大语言模型（Claude claude-opus-4-6）的自动化光学镜头设计系统。用户用自然语言描述镜头需求，Agent 自动完成初始结构设计、参数优化和性能评估的完整流程。

## 功能特性

- **自然语言输入**：支持中文和英文描述镜头需求
- **自动结构设计**：根据焦距、光圈、视场角自动选择合适的镜头结构（单片、双合、三片、双高斯、长焦、广角）
- **梯度优化**：调用 [DeepLens](https://github.com/singer-yang/DeepLens) 框架进行可微分光线追迹优化
- **性能评估**：输出 RMS 弥散斑大小、畸变、色差、MTF 等指标
- **结构迭代**：根据评估结果智能决策是否增减镜片（最多 3 次结构修改）
- **结果导出**：保存为 JSON 格式或 Zemax `.zmx` 格式

## 快速开始

### 1. 安装依赖

```bash
pip install anthropic torch numpy matplotlib scipy
# 安装 DeepLens（用于光学仿真和优化）
pip install git+https://github.com/singer-yang/DeepLens.git
```

### 2. 设置 API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. 运行 Agent

```bash
# 交互模式
python agent.py

# 直接传入描述（中文）
python agent.py --description "设计一个50mm f/1.8全画幅相机标准镜头"

# 直接传入描述（英文）
python agent.py --description "Design a 24mm f/2.8 wide-angle lens for APS-C sensor"

# 允许更多结构迭代次数
python agent.py --description "telephoto 200mm f/4 with minimal chromatic aberration" --max-iter 5
```

## 工作流程

```
用户输入（自然语言）
    ↓
解析需求（焦距、光圈、FOV、传感器尺寸）
    ↓
选择初始结构模板 → 缩放到目标参数
    ↓
DeepLens 梯度优化（2000次迭代）
    ↓
性能评估（RMS、畸变、色差）
    ↓
判断是否需要修改结构？
  ├── 是 → 增减镜片 → 重新优化 → 重新评估
  └── 否 → 保存结果
    ↓
生成设计报告
```

## 镜头模板库

| 模板 | 适用场景 | 元件数 | 示例 |
|------|---------|--------|------|
| `singlet` | 长焦、慢速、简单 | 1 | 100mm f/10 |
| `doublet` | 消色差、中等焦距 | 2 | 50mm f/4 |
| `triplet` | 中等视场、中等光圈 | 3 | 50mm f/2.8 |
| `double_gauss` | 大光圈标准镜 | 6 | 50mm f/1.8 |
| `telephoto` | 长焦压缩比例 | 5 | 200mm f/4 |
| `wide_angle` | 广角（FOV > 70°） | 6 | 24mm f/2.8 |

## 项目结构

```
optical_agent/
├── agent.py            # 主 Agent（Claude tool use 主循环）
├── lens_tools.py       # 工具实现（DeepLens 封装）
├── lens_templates.py   # 模板选择与缩放逻辑
├── templates/          # 初始镜头结构 JSON
│   ├── singlet.json
│   ├── doublet.json
│   ├── triplet.json
│   ├── double_gauss.json
│   ├── telephoto.json
│   └── wide_angle.json
├── results/            # 优化结果输出目录
├── requirements.txt
└── README.md
```

## Agent 工具说明

| 工具 | 功能 |
|------|------|
| `design_initial_structure` | 从规格生成初始镜头 JSON |
| `run_optimization` | 运行 DeepLens 梯度优化 |
| `evaluate_lens` | 提取性能指标 |
| `add_lens_element` | 在指定位置插入新镜片 |
| `remove_lens_element` | 移除指定镜片 |
| `save_lens` | 保存为 JSON/ZMX 格式 |
| `generate_report` | 生成设计报告 |

## 注意事项

- **DeepLens 未安装时**：Agent 仍可运行，但优化和评估会使用模拟数据（标注为 `[MOCK]`）
- **GPU 加速**：如有 CUDA GPU，DeepLens 将自动使用 GPU 加速优化（推荐）
- **优化时间**：CPU 上 2000 次迭代约需 5-15 分钟；GPU 上约需 1-3 分钟
- **结果目录**：优化结果保存在 `results/` 目录，包含优化过程图表和最终镜头参数

## 依赖

- `anthropic >= 0.40.0`：Claude API
- `torch >= 2.0.0`：PyTorch（DeepLens 后端）
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `scipy >= 1.10.0`
- [DeepLens](https://github.com/singer-yang/DeepLens)（光学仿真框架）
