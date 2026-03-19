# 工业设备运维诊断 Agent Skill

## 基本信息

```
skill_name: industrial_equipment_ops_diagnosis
version: 1.0.0
author: AI Agent Team
description: 面向工业场景，对接传感器与设备监测数据，实现故障自动诊断、根因分析与运维工单生成
```

## 触发短语

- 设备故障诊断
- 工业设备运维
- 传感器异常分析
- 设备故障工单

## 权限

- `read:industrial_sensor_data` - 读取工业传感器数据
- `read:equipment_knowledge_base` - 读取设备知识库
- `write:maintenance_work_order` - 写入运维工单

## 依赖

```python
torch>=2.0.0
numpy>=1.24.0
```

## 核心职责与场景边界

- **单一职责**：仅处理工业设备（如机床、风机、生产线）的运维诊断，不涉及交通、低空等其他场景
- **场景边界**：
  - 输入：设备实时/历史传感器数据（振动、温度、压力、电流等）
  - 输出：故障定位、根因分析、运维建议及可执行工单
- **禁止行为**：不处理非工业设备数据、不生成超出运维范畴的指令

## 流程拆解

### Step 1: 数据采集与校验
- 工具调用：`industrial_data_api.get_sensor_data(equipment_id, time_range)`
- 校验规则：缺失值>10%则终止，返回数据异常提示
- 输出：传感器数据

### Step 2: 异常检测
- 工具调用：`tf_saved_model.load("anomaly_detection_model").predict(data)`
- 模型：1D-CNN + Bi-LSTM + Attention
- 输出：异常片段、异常指标、置信度

### Step 3: 根因推理
- 工具调用：`equipment_knowledge_base.query(anomaly_type, equipment_model)`
- 输出：故障根因、影响范围、历史同类案例

### Step 4: 运维建议与工单生成
- 工具调用：`work_order_api.create(equipment_id, fault_info, priority)`
- 输出：工单ID、处理优先级、推荐操作步骤

## 输入输出规范

### 输入示例
```json
{
  "equipment_id": "MACHINE-001",
  "time_range": ["2025-03-19T00:00:00", "2025-03-19T12:00:00"],
  "data_type": ["vibration", "temperature", "current"]
}
```

### 输出示例
```json
{
  "status": "success",
  "anomaly": {
    "type": "bearing_fault",
    "confidence": 0.92,
    "severity": "high",
    "time_segment": "2025-03-19T08:15:00 ~ 08:30:00",
    "affected_sensors": ["vibration", "temperature"]
  },
  "root_cause": "轴承磨损导致振动超标，关联温度升高",
  "impact": "可能导致生产线停机，影响范围为A区3号产线",
  "maintenance_suggestions": [
    "立即停机检查轴承状态",
    "更换磨损轴承并加注润滑脂",
    "重启后监测2小时数据"
  ],
  "work_order": {
    "id": "WO-20250319-001",
    "priority": "high",
    "assignee": "maintenance_team_A",
    "estimated_duration": "2-4小时"
  }
}
```

## 支持的故障类型

| ID | 类型 | 类别 |
|----|------|------|
| 0 | normal | 正常 |
| 1 | bearing_fault | 机械 |
| 2 | gear_wear | 机械 |
| 3 | motor_failure | 电气 |
| 4 | temperature_overheat | 温度 |
| 5 | vibration_exceed | 振动 |
| 6 | pressure_anomaly | 压力 |
| 7 | current_fluctuation | 电气 |
| 8 | lubrication_failure | 工艺 |
| 9 | misalignment | 机械 |
| 10 | imbalance | 机械 |
| 11 | looseness | 机械 |
| 12 | cooling_failure | 温度 |
| 13 | power_supply_issue | 电气 |
| 14 | flow_blockage | 工艺 |
| 15 | leakage | 工艺 |
| 16 | resonance | 振动 |

## 优化与异常处理

### 性能优化
- **渐进式加载**：仅在检测到异常时加载根因推理模块
- **缓存**：常用设备型号的知识库条目缓存
- **模型集成**：多模型投票提升鲁棒性

### 异常处理

| 状态 | 场景 | 返回 |
|------|------|------|
| `data_error` | 数据缺失率>10% | `{"status": "data_error", "message": "传感器数据缺失率过高，请检查采集链路"}` |
| `inference_error` | 模型推理失败 | `{"status": "inference_error", "message": "诊断模型暂时不可用，请人工排查"}` |
| `timeout` | 推理超时 | `{"status": "timeout", "message": "诊断超时，请稍后重试"}` |

## 使用方法

```python
from skill import create_agent

# 创建 Agent
agent = create_agent(
    model_paths=["checkpoints/best_model.pt"],
    device="cuda"
)

# 执行诊断
result = agent.diagnose(
    equipment_id="MACHINE-001",
    time_range=["2025-03-19T00:00:00", "2025-03-19T12:00:00"],
    data_type=["vibration", "temperature", "current"]
)

print(result.status)
print(result.anomaly)
print(result.work_order)
```

## 训练

```bash
# 生成模拟数据
python train.py --generate_data --num_samples 10000

# 训练模型
python train.py --data_path data/sensor_data.json --epochs 50 --use_amp
```
