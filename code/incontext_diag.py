import json,sys,torch
sys.path.insert(0,'code')
from common import MODELS, load_facts
from transformers import AutoModelForCausalLM, AutoTokenizer
spec=MODELS['llama']
facts={f['id']:f for f in load_facts(spec['facts_v2'])}
recs=json.load(open('analysis/run_20261020_incontext_10200339_llama/records.json'))
tok=AutoTokenizer.from_pretrained(spec['dir']); m=AutoModelForCausalLM.from_pretrained(spec['dir'],torch_dtype=torch.bfloat16).eval()
for r in recs[:6]:
    f,new=facts[r['fact']],facts[r['new_fact']]
    for lg in ('en','es'):
        L=f['lang'][lg]; tgt=new['lang'][lg]['target']
        ctx=f"Fact: {f['lang']['en']['prompts'][0]} {new['lang']['en']['target']}.\n"
        for name,p in (('ctx',ctx+L['prompts'][0]),('plain',L['prompts'][0])):
            ids=tok(p,return_tensors='pt')
            with torch.no_grad(): o=m.generate(**ids,max_new_tokens=10,do_sample=False)
            print(lg,name,'| tgt=',tgt,'| gen=',repr(tok.decode(o[0,ids['input_ids'].shape[1]:])),flush=True)
