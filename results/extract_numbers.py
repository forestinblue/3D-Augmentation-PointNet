import csv, collections, json
R='4_evaluation_ablation/test_result_csv/test_result_exp{}.csv'
C="airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()
out={}
for i in range(1,7):
    rows=[];meta={}
    for r in csv.reader(open(R.format(i))):
        if not r or r[0]=='id': continue
        if r[0] in('Accuracy','F1_score'): meta[r[0]]=float(r[1]); continue
        rows.append((int(r[1]),int(r[2])))
    tot=collections.Counter(t for t,_ in rows); ok=collections.Counter(t for t,p in rows if t==p)
    # macro F1
    f1=[]
    for c in range(40):
        tp=ok[c]; fp=sum(1 for t,p in rows if p==c and t!=c); fn=tot[c]-tp
        f1.append(0 if tp==0 else 2*tp/(2*tp+fp+fn))
    out[i]=dict(n=len(rows),acc=sum(ok.values())/len(rows),f1=sum(f1)/40,csv=meta,
                percls={C[c]:[ok[c],tot[c]] for c in range(40)})
logs={}
for i,d in enumerate(['exp1_baseline','exp2_traditional','exp3_pointE','exp4_combined_traditional','exp5_combined_pointE','exp6_all_combined'],1):
    last=list(csv.reader(open(f'Data/{d}/train_log_exp{i}.csv')))[-1]
    logs[i]=dict(epoch=int(last[0]),loss=float(last[1]),acc=float(last[2]))
json.dump(dict(eval=out,logs=logs),open('results/numbers.json','w'),indent=1)
