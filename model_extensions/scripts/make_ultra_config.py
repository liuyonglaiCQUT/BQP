import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['WN18RR', 'FB15k237'], required=True)
p.add_argument('--mode', choices=['eval', 'finetune'], required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert a.out.resolve().is_relative_to(ROOT)
# Preserve official architecture/task/optimizer; render every template explicitly.
text = (ROOT / 'repos/ULTRA/config/transductive/inference.yaml').read_text()
values = {'dataset': a.dataset, 'gpus': '[0]', 'epochs': '0' if a.mode == 'eval' else '10',
          'bpe': 'null', 'ckpt': (ROOT / 'repos/ULTRA/ckpts/ultra_3g.pth').as_posix()}
for k, v in values.items():
    text = text.replace('{{ ' + k + ' }}', v)
cfg = yaml.safe_load(text)
cfg['output_dir'] = str(a.out.parent / 'outputs')
cfg['dataset']['root'] = str(ROOT / 'data/ULTRA')
a.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
