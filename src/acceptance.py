"""Run the stage-A gate and save a truthful machine-readable result."""
import argparse
import io
import sys
import unittest
from pathlib import Path
from data import read_config,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read_config(a.config)
    root=Path(c['root']);coverage=read_config(root/'reports/selection_coverage.json')
    log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (root/'reports/acceptance.log').write_text(log.getvalue());print(log.getvalue(),flush=True)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),required_checks=c['acceptance'],train_targets=c['dataset']['expected']['train']['targets'],validation_targets=c['dataset']['expected']['validation']['targets'],test_evaluated=False)
    write_json(root/'reports/acceptance.json',report)
    sys.exit(0 if result.wasSuccessful() else 1)
