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

| 顺序 | 工作 | 状态（2026-10-15 核对） |
|---|---|---|
| A | 训练／验证数据重建与全部实现验收 | 已完成，15 项验收通过 |
| 校准 | 分任务在 D00/D11 上进行三点学习率搜索，再做 D00 三种子阈值校准 | 已完成 |
| B | 16 个信息组合及 F+Z | 已完成 |
| C | 历史长度与三条关系规则的固定数量选择 | 已完成 |
| D | 连接 × 偏置完整 2×2 | 已完成 |
| 确认 | 补齐预登记条件的三种子结果 | 已完成 |
| E | 达到采用标准后进行跨模型验证 | 未启动，见采用判据报告 |
| 测试 | 配置全部锁定后统一复核 | 未开放 |

用户追加授权：协议与配置完成后，执行数据重建并启动训练。所有正式训练必须先通过阶段 A。旧代码、数据、结果和检查点不修改；本目录的模型从头训练。

状态依据 [队列状态](reports/queue_status.json) 和 [队列日志](reports/queue.log)。队列完成不代表所有条件已经收敛；各运行另记停止原因。测试集已有历史使用记录，最终评估应标记为复核。

## 当前产物

- [数据重建数量](reports/data_counts.json)
- [坐标系核验](reports/coordinate_audit.json)
- [C2 覆盖率、q 分布及匹配误差](reports/selection_coverage.json)
- [验收结果](reports/acceptance.json) 与 [完整验收日志](reports/acceptance.log)
- [学习率预实验清单](reports/lr_search_plan.json)
- [运行说明](RUNNING.md)

已生成 `selected_learning_rates.json`、`thresholds.json`、`formal_plan.json`、[阶段结果](reports/stage_results.json)与[跨模型采用判据](reports/external_gate.json)。本协议的阶段 E 未自动启动，测试集未评估；下列追加实验采用各自独立协议。

## 历史终点追加实验

按用户追加要求，已在 [history_endpoints_v1](history_endpoints_v1/README.md) 完成一个独立 ETP 条件的三个 seed：固定 `C2_current_phase_control`，加入历史有效终点，只屏蔽锚点终点；Raw10 与 Pass 标签不变。后续复核发现成对对抗中的历史终点泄露，本轮结果已标记无效，不能作为性能提升证据。原实验产物保留用于审计。

## 合并对抗后的 5 任务 HGT（2026-10-14）

历史终点追加实验经复核为泄露（见其 README 顶部说明）。按用户要求，成对对抗合并为一个事件，在干净信息集上运行 5 任务 HGT 并逐项检验关系作用，见 [five_task_hgt_v1](five_task_hgt_v1/README.md)。

42 个正式运行已完成；验证集三种子结果见 [汇总表](five_task_hgt_v1/reports/tables.md)。测试集未评估。

## 下一执行球员的最小信号筛查（2026-10-14）

检验动态球员历史是否有独立于静态身份、频率和最近出现的信息，以及是否需要候选条件化注意力，见 [signal_screening_v1](signal_screening_v1/README.md)。

- [阶段 1](signal_screening_v1/reports/REPORT.md)：18 个正式运行已完成。
- [阶段 2：固定学习率日程](signal_screening_v1/reports_stage2/REPORT.md)：15 个运行已完成。
- [阶段 3a：本场更长历史](signal_screening_v1/reports_stage3a/REPORT.md)：6 个运行已完成。
- 阶段 3b 的球员间关系代码已实现，尚未启动。测试集未评估。
