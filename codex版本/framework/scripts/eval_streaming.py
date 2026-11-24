"""Paired document-level evaluation with current-chunk-only attention input."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lact_qwen.loading import add_model_args, load_model, autocast
from lact_qwen.streaming import begin_stream, forward_chunk, end_stream, state_bytes


def evaluate(model, tokens, mode, chunk, context):
    docs=[]
    for i, row in enumerate(tokens):
        ids=torch.tensor(np.array(row), device=next(model.parameters()).device, dtype=torch.long)[None]
        donor=torch.tensor(np.array(tokens[(i+1)%len(tokens)]), device=ids.device, dtype=torch.long)[None]
        begin_stream(model, chunk)
        donor_state={}
        bins=[];sizes=[]
        with torch.no_grad(), context():
            for s in range(0, ids.shape[1], chunk):
                e=min(s+chunk, ids.shape[1]);last=min(e, ids.shape[1]-1)
                if mode=='wrong':
                    model.lact_ctx.states=donor_state
                logits=forward_chunk(model, ids[:,s:e], s, update=(mode=='write'), reset=(mode=='reset'))
                loss=F.cross_entropy(logits[0,:last-s].float(), ids[0,s+1:last+1], reduction='none')
                bins.append({'start':s,'end':last,'loss':loss.double().mean().item(),
                             'correct':int((logits[0,:last-s].argmax(-1)==ids[0,s+1:last+1]).sum()),'tokens':last-s})
                del logits,loss
                if mode=='wrong':
                    model.lact_ctx.states=donor_state
                    donor_logits=forward_chunk(model,donor[:,s:e],s,update=True)
                    donor_state=model.lact_ctx.states
                    del donor_logits
                sizes.append(state_bytes(model.lact_ctx.states))
        # LaCT allocates momentum on the first write, then keeps fixed size.
        if len(set(sizes[1:]))>1:
            raise AssertionError('Memory state grows with history: '+str(sizes))
        end_stream(model)
        metrics={'last2k':bins[-1]['loss'],
                 'beyond2k':sum(x['loss']*x['tokens'] for x in bins[1:])/sum(x['tokens'] for x in bins[1:]),
                 'last2k_token_accuracy':bins[-1]['correct']/bins[-1]['tokens']}
        docs.append({'document':i,'token_sha256':hashlib.sha256(np.asarray(row,dtype=np.int32).tobytes()).hexdigest(),
                     'bins':bins,'metrics':metrics,'persistent_state_bytes':sizes})
        print(f'{mode} document {i+1}/{len(tokens)} last2k={metrics["last2k"]:.6f}',flush=True)
    return docs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',required=True)
    p.add_argument('--mode',choices=['write','nowrite','reset','wrong'],required=True)
    p.add_argument('--num-docs',type=int,default=48)
    p.add_argument('--out',required=True)
    add_model_args(p);args=p.parse_args()
    model,cfg=load_model(args)
    tokens=np.load(args.tokens,mmap_mode='r')[:args.num_docs]
    assert len(tokens)==args.num_docs and len(tokens)>1 and tokens.shape[1]%cfg.chunk_size==0
    docs=evaluate(model,tokens,args.mode,cfg.chunk_size,lambda:autocast(args))
    result={'args':vars(args),'docs':docs,'input_chunk':cfg.chunk_size,'no_attention_kv_cache':True,
            'tokens_sha256':hashlib.sha256(Path(args.tokens).read_bytes()).hexdigest(),
            'mean':{k:float(np.mean([d['metrics'][k] for d in docs])) for k in docs[0]['metrics']},
            'trainable_memory_parameters':sum(p.numel() for n,p in model.named_parameters() if '.memory.' in n)}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(out)
if __name__=='__main__':main()
