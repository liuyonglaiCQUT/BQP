"""Run upstream preprocessing with a one-worker thread pool.

The upstream functions are executed from this wrapper and therefore are not
importable members of the process' real ``__main__`` module. A process Pool
would try to pickle them by that name and fail on some Linux/Python builds.
One preprocessing worker is already requested, so a ThreadPool preserves the
result while avoiding cross-process pickling.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
task = sys.argv[1]
assert task in ['WN18RR', 'FB15k237', 'NELL995']
repo = ROOT / 'repos/SimKGC'
data = repo / 'data' / task

if task == 'NELL995':
    source = (ROOT / 'scripts' / 'prepare_simkgc_nell995.py').read_text(encoding='utf-8')
    exec(compile(source, str(ROOT / 'scripts' / 'prepare_simkgc_nell995.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts' / 'prepare_simkgc_nell995.py')})
    raise SystemExit(0)

# A repository copied through a Windows checkout may contain CRLF shell files.
# Normalize only upstream launchers; Python accepts either line ending.
for launcher in (repo / 'scripts').glob('*.sh'):
    raw = launcher.read_bytes()
    normalized = raw.replace(b'\r\n', b'\n')
    if normalized != raw:
        launcher.write_bytes(normalized)

sys.argv = ['preprocess.py', '--task', task, '--workers', '1',
            '--train-path', str(data / 'train.txt'), '--valid-path', str(data / 'valid.txt'),
            '--test-path', str(data / 'test.txt')]
source = (repo / 'preprocess.py').read_text(encoding='utf-8')
source = source.replace('from multiprocessing import Pool',
                        'from multiprocessing.pool import ThreadPool as Pool')
source = source.replace("mp.set_start_method('fork')", '')
exec(compile(source, str(repo / 'preprocess.py'), 'exec'), {'__name__': '__main__'})
