# P18 Second Model + P15 AG News — Kaggle T4
# =============================================
# P18: Leave-one-out AG News with 0.5B (5 seeds) — prove 0% isn't model-specific
# P15: Multi-task AG News with LoRA (5 seeds)
# Total: ~5h on T4

# Cell 1: Setup
!pip install -q transformers>=4.45 peft>=0.13 accelerate>=0.34 datasets scikit-learn

import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/kaggle/working/q1_p18_p15"
os.makedirs(SAVE, exist_ok=True)

def autosave(r, fn):
    with open(f"{SAVE}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{SAVE}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ {fn} | F1={r.get('strict_f1','?')}")

# Cell 2: Generic train+eval
def run_exp(model_name, dataset, seed, paper, rank=64,
            train_n=5000, test_n=500, epochs=3, lr=2e-4,
            batch=4, gacc=8, ml=512, train_classes=None, test_classes=None):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 {paper} | {model_name.split('/')[-1]} | {dataset} | s={seed}")
    if train_classes: print(f"  Train classes: {train_classes}")
    if test_classes: print(f"  Test classes: {test_classes}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(model_name,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,}")

    labels = ["World","Sports","Business","Sci/Tech"]
    ds_raw = load_dataset("ag_news",split="train").shuffle(seed=seed)
    test_raw = load_dataset("ag_news",split="test").shuffle(seed=seed)

    # Filter by class if specified (leave-one-out)
    if train_classes is not None:
        ds_raw = ds_raw.filter(lambda x: x["label"] in train_classes)
        test_raw = test_raw.filter(lambda x: x["label"] in (test_classes if test_classes else train_classes))
    ds_raw = ds_raw.select(range(min(train_n, len(ds_raw))))
    test_raw = test_raw.select(range(min(test_n, len(test_raw))))

    train_pairs = [(f"Classify this news: {e['text'][:300]}", labels[e['label']]) for e in ds_raw]
    test_pairs = [(f"Classify this news: {e['text'][:300]}", labels[e['label']]) for e in test_raw]

    # Tokenize
    ai,aa,al = [],[],[]
    for pr,resp in train_pairs:
        msgs=[{"role":"user","content":pr},{"role":"assistant","content":resp}]
        s=tok.apply_chat_template(msgs,tokenize=False)
        enc=tok(s,truncation=True,max_length=ml,padding="max_length")
        ids=enc["input_ids"];at=enc["attention_mask"]
        ai.append(ids);aa.append(at);al.append([l if a==1 else -100 for l,a in zip(ids,at)])
    ds=Dataset.from_dict({"input_ids":ai,"attention_mask":aa,"labels":al})
    ds.set_format("torch")
    loader=DataLoader(ds, batch_size=batch, shuffle=True)
    opt=torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0;opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc;loss.backward();tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step();sched.step();opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    # Eval
    mdl.eval(); ps,gs=[],[]
    for pr,gold in test_pairs:
        msgs=[{"role":"user","content":pr}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=20,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ps.append(resp.split("\n")[0].strip());gs.append(gold)
        if len(ps)%100==0: print(f"  eval {len(ps)}/{len(test_pairs)}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":paper,"model":model_name.split("/")[-1],"dataset":dataset,
       "seed":seed,"strict_f1":round(f1,4),"accuracy":round(acc,4),
       "train_min":round(tm,1),"params":params,
       "train_classes":str(train_classes),"test_classes":str(test_classes),
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    fn=f"{paper}_{model_name.split('/')[-1]}_{dataset}_s{seed}.json"
    autosave(r, fn)
    del mdl,tok; torch.cuda.empty_cache()
    return r

# Cell 3: P18 — Leave-one-out (train on 3 classes, test on held-out class)
ALL = []
print("🔴"*30 + "\n[P18] Leave-one-out AG News — 0.5B")
labels_map = {0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}

for held_out in range(4):
    train_cls = [i for i in range(4) if i != held_out]
    for s in [42, 77, 123, 456, 999]:
        r = run_exp("Qwen/Qwen2.5-0.5B-Instruct", "ag_news_loo", s,
                    f"P18_hold{held_out}",
                    train_classes=train_cls, test_classes=[held_out],
                    train_n=3750, test_n=250)
        ALL.append(r)

# Cell 4: P15 — AG News standard (5 seeds)
print("\n" + "🟢"*30 + "\n[P15] AG News standard — 5 seeds")
for s in [42, 77, 123, 456, 999]:
    ALL.append(run_exp("Qwen/Qwen2.5-0.5B-Instruct", "ag_news", s, "P15"))

# Cell 5: Summary
print("\n" + "="*70)
print("📊 ALL RESULTS")
print("="*70)
for r in ALL:
    print(f"  {r['paper']:<15} s={r['seed']:<4} F1={r['strict_f1']}")
print(f"\nSaved: {SAVE}/summary.csv")
print("🏁 DONE")
