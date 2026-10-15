# 下一执行球员：最小信号筛查

只回答三个问题：动态球员历史是否有独立于静态身份、频率和最近出现的信息？是否需要候选条件化注意力才能利用它？是否值得继续研究更长历史和实体关系？另加一个序列基线 M0_ctx_identity（球员身份挂在事件上，候选只有静态表示）。完整协议见 [PROTOCOL.md](PROTOCOL.md)，配置见 [configs/experiment.json](configs/experiment.json)。

阶段 2（[configs/stage2_fixed_schedule.json](configs/stage2_fixed_schedule.json)）用所有运行相同的固定学习率日程重跑 M1、M2、M2_shuffled、M0_ctx_identity，并加入点积打分的序列基线 M0_ctx_identity_dot；结果在 `reports_stage2/`。

阶段 3a（[configs/stage3a_match_history.json](configs/stage3a_match_history.json)）检验窗口外的本场历史（H_match、H_match_shuffled），参照为阶段 2 的 M2；结果在 `reports_stage3a/`。3b（球员间关系）的代码已就绪，等待确定基准条件。

- `src/scr_candidates.py`：在场候选所需的时间数据（首次出手、换人分钟、红牌），以及按锚点计算候选集合。
- `src/scr_batches.py`：情境输入（M0_ctx_identity 另带每个单元的参与名单槽位）和逐候选的输入（静态、频率与最近出现、本人历史、重排对照）。
- `src/scr_models.py`：情境编码器和 M0–M3 的候选编码器；M0_ctx_identity 把参与球员的静态表示加到事件 token 上。
- `src/scr_scoring.py`、`src/scr_fit.py`、`src/scr_queue.py`、`src/scr_report.py`：评分、训练、调度、报告。
- `tests/test_screening.py`：25 项验收（结果见 `reports/acceptance.json`）。
- `archive/before_context_identity_20261014/`：第一波 8 个正式运行所用的代码和当时的验收结果。
- `archive/stage1_final_20261015/`：阶段 1 结束时的代码、配置和验收结果。
- `archive/stage2_final_20261015/`：阶段 2 运行所用的代码、配置和验收结果。
- `analysis/stage2_checks.py`：阶段 2 的重跑在日程分叉之前与阶段 1 逐位一致、学习率按日程变化，以及每个运行的 top-1 快照。
- `analysis/posthoc_contrasts.py`：事后对照（不属于预注册判定）和去掉 seed 20260716 的稳健性表，写入 `reports/posthoc_contrasts.json`。该 seed 的 M1_summary 在"身份跳变"之前两次衰减学习率，跳变没有发生。
- 数据只读复用 `../five_task_hgt_v1/data`；`data/` 只保存候选时间数据。

```bash
cd /home/li/football/graph_prediction_research/signal_screening_v1
PY=/home/li/football/paper_1/Phase_1/.venv/bin/python
$PY -B src/scr_candidates.py --config configs/experiment.json   # 已完成
$PY -B tests/test_screening.py                                  # 已通过（25 项）
$PY -B src/scr_queue.py --config configs/experiment.json        # 已运行：LR 搜索和第一波 8 个正式运行
$PY -B src/scr_queue.py --config configs/experiment.json --resume-formal configs/resume_formal_20261014.json   # 已完成：其余 10 个和报告
$PY -B src/scr_queue.py --config configs/stage2_fixed_schedule.json --formal-only   # 阶段 2，已完成
$PY -B src/scr_queue.py --config configs/stage3a_match_history.json --formal-only    # 阶段 3a，已完成
$PY -B analysis/stage2_checks.py --stage1 configs/experiment.json --stage2 configs/stage2_fixed_schedule.json
cat reports/queue_status.json; tail -n 20 reports/queue.log
```

结果写入 `reports/REPORT.md` 和 `reports/summary.json`。测试集不读取、不评估。
