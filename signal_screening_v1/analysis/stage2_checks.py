"""Stage-2 checks: reruns repeat stage 1 exactly until the schedules part, the learning rate follows the fixed schedule,
and per-seed top-1 snapshots show when each run's identity jump happened. Works on finished or running runs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scr_fit import fixed_step_lr

COMPARED = ('optimizer_steps', 'lr', 'train_loss', 'validation')
SNAPSHOTS = (48, 64, 80, 96, 108, 120)


def curve(run):
    return {r['epoch']: r for r in map(json.loads, (run / 'curve.jsonl').open())}


def main(stage1, stage2):
    lr = stage2['learning_rate']['value']
    schedule = stage2['training']['schedule']
    runs1, runs2 = Path(stage1['paths']['runs']), Path(stage2['paths']['runs'])
    out = []
    for seed in stage2['training']['seeds']:
        for c in stage2['conditions']:
            run = runs2 / f'formal_{c}_s{seed}'
            if not (run / 'curve.jsonl').exists():
                continue
            rows = curve(run)
            item = dict(run=run.name, epochs=max(rows), status=json.load(open(run / 'status.json'))['status'],
                        lr_follows_schedule=all(r['lr'] == fixed_step_lr(schedule, lr, e) for e, r in rows.items()),
                        top1={e: round(100 * rows[e]['validation']['single_top1'], 2) for e in SNAPSHOTS if e in rows})
            if c in stage1['conditions']:
                old = curve(runs1 / f'formal_{c}_s{seed}')
                decays = [e for e, r in old.items() if r['scheduler_action'] == 'decay']
                # Stage 1 changed the learning rate after its first decay epoch; stage 2 after epoch 96.
                same_until = min(decays + [schedule['milestones'][0]])
                checked = [e for e in rows if e <= same_until]
                item.update(stage1_first_decay=decays[0] if decays else None, expected_identical_through=same_until,
                            identical_so_far=all(all(rows[e].get(k) == old[e].get(k) for k in COMPARED) for e in checked),
                            checked_epochs=len(checked))
                if max(rows) > same_until:
                    e = same_until + 1
                    item['differs_after'] = any(rows[e].get(k) != old[e].get(k) for k in COMPARED)
            out.append(item)
    Path(stage2['paths']['reports'], 'stage2_checks.json').write_text(json.dumps(out, indent=2) + '\n')
    for item in out:
        rerun = '' if 'identical_so_far' not in item else (
            f"identical to stage 1 through epoch {item['expected_identical_through']}: {item['identical_so_far']} "
            f"({item['checked_epochs']} epochs checked){'; differs after: ' + str(item['differs_after']) if 'differs_after' in item else ''}")
        print(f"{item['run']:38} {item['status']:8} epoch {item['epochs']:3}  lr ok {item['lr_follows_schedule']}  {rerun}")
        print(' ' * 39 + 'top-1 ' + '  '.join(f'e{e} {v}' for e, v in item['top1'].items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1', required=True)
    parser.add_argument('--stage2', required=True)
    a = parser.parse_args()
    main(json.load(open(a.stage1)), json.load(open(a.stage2)))
