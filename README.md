# 足球事件预测实验

研究固定信息量下，信息内容、历史选择、关系连接和关系偏置对下一条足球事件预测的影响，并在满足采用标准后进行跨模型验证。

## 协议与配置

- [完整实验协议](PROTOCOL.md)
- [实验配置](configs/experiments.json)
- `data/`：重新构建的白名单输入、独立目标及历史选择。
- `runs/`：独立训练运行、配置、学习曲线、检查点与预测。
- `reports/`：验收、学习率选择、阶段报告与运行状态。
- `src/`、`tests/`：本轮新增实现与验收测试。

## 执行顺序

| 顺序 | 工作 | 初始状态 |
|---|---|---|
| A | 训练／验证数据重建与全部实现验收 | 已完成，15 项验收通过 |
| 校准 | 分任务在 D00/D11 上进行三点学习率搜索，再做 D00 三种子阈值校准 | 运行中 |
| B | 16 个信息组合及 F+Z | 已排队，等待校准 |
| C | 历史长度与三条关系规则的固定数量选择 | 已排队，等待校准 |
| D | 连接 × 偏置完整 2×2 | 已排队，等待校准 |
| 确认 | 补齐预登记条件的三种子结果 | 待执行 |
| E | 达到采用标准后进行跨模型验证 | 等待门槛 |
| 测试 | 配置全部锁定后统一复核 | 未开放 |

用户追加授权：协议与配置完成后，执行数据重建并启动训练。所有正式训练必须先通过阶段 A。旧代码、数据、结果和检查点不修改；本目录的模型从头训练。

表格记录本次启动时的状态；实时进度见 [队列状态](reports/queue_status.json) 和 [队列日志](reports/queue.log)。测试集已有历史使用记录，最终评估应标记为复核。

## 当前产物

- [数据重建数量](reports/data_counts.json)
- [坐标系核验](reports/coordinate_audit.json)
- [C2 覆盖率、q 分布及匹配误差](reports/selection_coverage.json)
- [验收结果](reports/acceptance.json) 与 [完整验收日志](reports/acceptance.log)
- [学习率预实验清单](reports/lr_search_plan.json)
- [运行说明](RUNNING.md)

后续按流程生成 `selected_learning_rates.json`、`thresholds.json`、`formal_plan.json`、阶段报告与 `external_gate.json`。队列完成 B/C/D 后输出跨模型采用判据，不提前训练 HGT/GRU，不自动开启测试集。

## 历史终点追加实验

按用户追加要求，已在 [history_endpoints_v1](history_endpoints_v1/README.md) 实现一个独立 ETP 条件：固定当前最好组 `C2_current_phase_control`，加入历史有效终点，只屏蔽锚点终点；Raw10 与 Pass 标签不变。数据已构建、8 项验收通过，后台等待原队列成功结束后执行三个 seed。原实验协议、配置、代码和结果不变。

## 合并对抗后的 5 任务 HGT（2026-10-14）

历史终点追加实验经复核为泄露（见其 README 顶部说明）。按用户要求，成对对抗合并为一个事件，在干净信息集上运行 5 任务 HGT 并逐项检验关系作用，见 [five_task_hgt_v1](five_task_hgt_v1/README.md)。
