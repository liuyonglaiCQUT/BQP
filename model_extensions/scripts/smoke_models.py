"""Offline functional tests, not benchmark-performance claims.
Windows --unfused tests ULTRA's official PyG reference path without nvcc.
Linux default also compiles and exercises the production rspmm kernel.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt':
    sys.path.insert(0, str(ROOT / 'local_packages'))
    os.environ['PYTHONPATH'] = str(ROOT / 'local_packages')
for name, relative in [('HF_HOME','cache/huggingface'),('TORCH_HOME','cache/torch'),
                       ('TORCH_EXTENSIONS_DIR','cache/torch_extensions'),('TMP','tmp'),('TEMP','tmp')]:
    path = ROOT / relative; path.mkdir(parents=True, exist_ok=True); os.environ[name] = str(path)
import torch


def compounde():
    sys.path.insert(0,str(ROOT/'adapters'))
    from compounde import CompoundE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(7)
    for variant in ['fb_complete_srt','wn_rst_st','ogb_tail_str']:
        m = CompoundE(7,2,8,6,variant).to(device)
        batch = torch.tensor([[0,0,1],[2,1,3]],device=device)
        for direction, col in [('head',0),('tail',2)]:
            scores = m(batch,torch.arange(7,device=device).expand(2,-1),direction)
            slow = []
            for c in range(7):
                b = batch.clone(); b[:,col]=c; slow.append(m(b).squeeze(1))
            torch.testing.assert_close(scores,torch.stack(slow,1))
            # Independent explicit matrix transform, including noncommuting order.
        def transform(x, s, tr, a, order, halves):
            x = torch.nn.functional.normalize(x,dim=-1)
            pairs = [(i,i+4) for i in range(4)] if halves else [(i,i+1) for i in range(0,8,2)]
            mat = torch.zeros(8,8,device=device)
            for k,(i,j) in enumerate(pairs):
                angle=a[k]*torch.pi/m.embedding_range; co,si=angle.cos(),angle.sin()
                mat[i,i]=co; mat[i,j]=-si; mat[j,i]=si; mat[j,j]=co
            for op in order:
                x = x+tr if op=='T' else s*x if op=='S' else mat@x
            return x
        refs=[]
        for h,r,t in batch:
            parts=m.relation(r).split(8)
            he,te=m.entity(h),m.entity(t)
            if variant=='ogb_tail_str':
                s,tr,a=parts
                he=torch.nn.functional.normalize(he,dim=-1)
                te=transform(te,s,tr,a,'RTS',False)
            else:
                sh,th,angle,st,tt=parts
                he=transform(he,sh,th,angle,'TRS' if variant=='fb_complete_srt' else 'TSR',variant=='wn_rst_st')
                te=transform(te,st,tt,angle,'TRS' if variant=='fb_complete_srt' else 'TS',variant=='wn_rst_st')
            refs.append(6-(he-te).abs().sum())
        torch.testing.assert_close(m(batch).squeeze(1),torch.stack(refs))
        m(batch).sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # Filtered rank checks with tied scores, other known truths, chunk invariance.
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_compounde import evaluate,filters
    import numpy as np
    m=CompoundE(7,2,8,6,'fb_complete_srt').to(device)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    rows=np.array([[0,0,1],[0,0,2]])
    known=filters(rows)
    ranks,_=evaluate(m,rows,known,2,3,device)
    # Tail: one other true answer excluded -> 6 candidates, average rank 3.5.
    # Head: all 7 candidates -> average rank 4.
    np.testing.assert_allclose(ranks,[[4,3.5],[4,3.5]])
    ranks2,_=evaluate(m,rows,known,1,7,device)
    np.testing.assert_allclose(ranks,ranks2)
    return {'device':device,'tests':['three variants vs explicit matrices','shared FB head/tail rotation','head/tail vectorization','finite backward','filtered ties and chunk invariance']}


def simkgc(out):
    from transformers import BertConfig,BertModel,BertTokenizer
    repo=ROOT/'repos/SimKGC'
    tiny=out/'tiny_bert'; tiny.mkdir()
    vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','entity','relation','inverse','description']+[str(i) for i in range(8)]
    (tiny/'vocab.txt').write_text('\n'.join(vocab),encoding='utf-8')
    BertTokenizer(str(tiny/'vocab.txt')).save_pretrained(tiny)
    BertModel(BertConfig(vocab_size=len(vocab),hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64)).save_pretrained(tiny)
    data=out/'data'; data.mkdir()
    entities=[{'entity_id':str(i),'entity':f'entity {i}','entity_desc':f'description {i}'} for i in range(8)]
    (data/'entities.json').write_text(json.dumps(entities))
    for split,offset in [('train',1),('valid',2),('test',3)]:
        rows=[{'head_id':str(i),'head':f'entity {i}','relation':'relation','tail_id':str((i+offset)%8),'tail':f'entity {(i+offset)%8}'} for i in range(8)]
        (data/(split+'.txt.json')).write_text(json.dumps(rows))
    env=os.environ.copy(); env['SIMKGC_GRAD_CHECKPOINT']='1'
    command=[sys.executable,str(repo/'main.py'),'--task','WN18RR','--model-dir',str(out/'checkpoint'),
             '--pretrained-model',str(tiny),'--train-path',str(data/'train.txt.json'),'--valid-path',str(data/'valid.txt.json'),
             '--batch-size','4','--epochs','1','--workers','0','--seed','7','--pooling','mean',
             '--pre-batch','2','--use-self-negative','--use-link-graph','--finetune-t','--use-amp']
    with (out/'train.log').open('w') as f:
        subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    checkpoint=out/'checkpoint/model_last.mdl'
    saved=torch.load(checkpoint,map_location='cpu')
    assert {'optimizer','scheduler','scaler','python_rng_state','torch_rng_state'} <= saved.keys()
    with (out/'resume.log').open('w') as f:
        subprocess.run(command+['--resume',str(checkpoint)],cwd=repo,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    assert 'at epoch 1/1' in (out/'resume.log').read_text()
    command=[sys.executable,str(repo/'evaluate.py'),'--task','WN18RR','--is-test',
             '--pretrained-model',str(tiny),
             '--eval-model-path',str(checkpoint),'--train-path',str(data/'train.txt.json'),
             '--valid-path',str(data/'test.txt.json'),'--neighbor-weight','0.05','--rerank-n-hop','5']
    with (out/'eval.log').open('w') as f:
        subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert len(list((out/'checkpoint').glob('eval_*json')))==2
    return {'device':'cuda','tests':['official trainer tiny BERT','AMP','gradient checkpointing','pre-batch and self negatives','complete-state epoch resume','save/reload','bidirectional full entity evaluation'],'benchmark_performance_verified':False}


def ultra(unfused):
    sys.path.insert(0,str(ROOT/'repos/ULTRA'))
    from ultra.models import Ultra
    from ultra.layers import GeneralizedRelationalConv
    from ultra import tasks
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing
    if unfused:
        # Local diagnostic only; never used by run_ultra.sh.
        GeneralizedRelationalConv.propagate=lambda self,edge_index,size=None,**kw: MessagePassing.propagate(self,edge_index,size,**kw)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    edge=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,0,0,1,2,3]])
    graph=Data(edge_index=edge,edge_type=torch.tensor([0,1,0,1,2,3,2,3]),num_nodes=4,num_relations=4)
    graph=tasks.build_relation_graph(graph).to(device)
    cfg={'input_dim':64,'hidden_dims':[64]*6,'message_func':'distmult','aggregate_func':'sum','short_cut':True,'layer_norm':True}
    model=Ultra({'class':'RelNBFNet',**cfg},{'class':'EntityNBFNet',**cfg}).to(device)
    state=torch.load(ROOT/'repos/ULTRA/ckpts/ultra_3g.pth',map_location=device)
    model.load_state_dict(state['model'],strict=True)
    batch=torch.tensor([[[0,1,0],[0,2,0]],[[2,3,1],[1,3,1]]],device=device)
    score=model(graph,batch)
    assert score.shape==(2,2) and torch.isfinite(score).all()
    score.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return {'device':device,'tests':['official full architecture','official 3g weights strict load','head/tail forward','backward'],'kernel':'official unfused reference' if unfused else 'compiled rspmm','benchmark_performance_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['compounde','simkgc','ultra'],required=True); p.add_argument('--unfused',action='store_true'); a=p.parse_args()
    out=ROOT/'smoke'/a.model/(time.strftime('%Y%m%d_%H%M%S')); out.mkdir(parents=True)
    try:
        result=compounde() if a.model=='compounde' else simkgc(out) if a.model=='simkgc' else ultra(a.unfused)
        result.update(status='passed',torch=torch.__version__,python=sys.version)
    except Exception as e:
        (out/'result.json').write_text(json.dumps({'status':'failed','error':repr(e)},indent=2)); raise
    (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
