"""Reproduce all model results: python train.py (Python 3.12)."""
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve)
from sklearn.calibration import calibration_curve
from core import ROOT, NUM, CAT, LINEAR_NUM, CHURN_FEATURES, LINEAR_FEATURES

def pipeline(nums, model):
    return Pipeline([('prep', ColumnTransformer([
        ('num', StandardScaler(), nums),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CAT)])), ('model', model)])

def class_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return { 'accuracy': float(accuracy_score(y,pred)), 'precision': float(precision_score(y,pred,zero_division=0)),
        'recall': float(recall_score(y,pred)), 'f1': float(f1_score(y,pred)), 'roc_auc': float(roc_auc_score(y,p)),
        'average_precision': float(average_precision_score(y,p)), 'brier': float(brier_score_loss(y,p)),
        'confusion_matrix': confusion_matrix(y,pred).tolist() }

def reg_metrics(y,p):
    return {'mae':float(mean_absolute_error(y,p)), 'rmse':float(np.sqrt(mean_squared_error(y,p))), 'r2':float(r2_score(y,p))}

def main():
    source = ROOT/'data/abc_customers.csv'
    d = pd.read_csv(source)
    assert d.shape == (5000,14) and d.isna().sum().sum() == 0
    assert d.customer_id.is_unique and set(d.churned.unique()) == {0,1}
    train, rest = train_test_split(d.index, test_size=.4, random_state=42, stratify=d.churned)
    val, test = train_test_split(rest, test_size=.5, random_state=42, stratify=d.loc[rest,'churned'])
    a,b,t = d.loc[train],d.loc[val],d.loc[test]
    choices=[]; fitted=[]
    for c in [.1, 1., 10.]:
        m=pipeline(NUM,LogisticRegression(C=c,max_iter=2000,random_state=42))
        m.fit(a[CHURN_FEATURES],a.churned)
        p=m.predict_proba(b[CHURN_FEATURES])[:,1]
        choices.append({'C':c,'validation_roc_auc':roc_auc_score(b.churned,p)})
        fitted.append(m)
    best=int(np.argmax([v['validation_roc_auc'] for v in choices])); log=fitted[best]
    vp=log.predict_proba(b[CHURN_FEATURES])[:,1]
    thresholds=np.round(np.arange(.10,.901,.01),2)
    scores=[f1_score(b.churned,vp>=th) for th in thresholds]
    threshold=float(sorted(zip(scores,thresholds), key=lambda x:(-x[0],abs(x[1]-.5)))[0][1])
    p=log.predict_proba(t[CHURN_FEATURES])[:,1]
    lin=pipeline(LINEAR_NUM,LinearRegression()).fit(a[LINEAR_FEATURES],a.watch_hours)
    rp=lin.predict(t[LINEAR_FEATURES]); vrp=lin.predict(b[LINEAR_FEATURES])
    # Split-conformal absolute-residual interval calibrated only on validation rows.
    level=min(1,np.ceil((len(b)+1)*.90)/len(b))
    radius=float(np.quantile(abs(b.watch_hours-vrp),level,method='higher'))
    baseline_reg=reg_metrics(t.watch_hours,np.repeat(a.watch_hours.mean(),len(t)))
    baseline_cls=class_metrics(t.churned,np.repeat(a.churned.mean(),len(t)),.5)
    stats=reg_metrics(t.watch_hours,rp)
    schema={'categories':{c:sorted(a[c].unique().tolist()) for c in CAT},
            'ranges':{c:[float(a[c].min()),float(a[c].max())] for c in NUM},
            'defaults':{c:float(a[c].median()) for c in NUM}}
    audit={'rows':len(d),'columns':len(d.columns),'missing_cells':int(d.isna().sum().sum()),
           'duplicate_rows':int(d.duplicated().sum()),'duplicate_ids':int(d.customer_id.duplicated().sum()),
           'churn_count':int(d.churned.sum()),'churn_rate':float(d.churned.mean()),
           'daily_hours_over_24':int((d.avg_watch_time_per_day>24).sum()),
           'derived_daily_match_fraction':float((abs(d.avg_watch_time_per_day-d.watch_hours/(d.last_login_days+1))<.011).mean()),
           'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    metrics={'version':'1.0','seed':42,'python':platform.python_version(),'sklearn':sklearn.__version__,
        'data_audit':audit,'split':{'train':len(a),'validation':len(b),'test':len(t)},'schema':schema,
        'selection':choices,'chosen_C':choices[best]['C'],'threshold':threshold,
        'churn_test':class_metrics(t.churned,p,threshold),'churn_test_at_050':class_metrics(t.churned,p,.5),
        'churn_validation':class_metrics(b.churned,vp,threshold),'churn_baseline':baseline_cls,
        'linear_test':stats,'linear_validation':reg_metrics(b.watch_hours,vrp),'linear_baseline':baseline_reg,
        'linear_interval_radius':radius,'linear_interval_test_coverage':float(((t.watch_hours>=np.maximum(0,rp-radius)) & (t.watch_hours<=rp+radius)).mean()),
        'linear_beats_mean_rmse':stats['rmse']<baseline_reg['rmse'],
        'training_mean_watch_hours':float(a.watch_hours.mean()),
        'target_note':'Snapshot churn label and recorded viewing hours; no validated future time horizon.'}
    for name,model in [('churn',log),('viewing',lin)]:
        joblib.dump(model,ROOT/f'models/{name}.joblib')
        coef=model.named_steps['model'].coef_.ravel()
        pd.DataFrame({'feature':model.named_steps['prep'].get_feature_names_out(),'coefficient':coef}).to_csv(ROOT/f'results/{name}_coefficients.csv',index=False)
    (ROOT/'results/metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'customer_id':d.customer_id,'split':['train' if i in set(train) else 'validation' if i in set(val) else 'test' for i in d.index]}).to_csv(ROOT/'results/split_membership.csv',index=False)
    pd.DataFrame({'customer_id':t.customer_id,'actual_churn':t.churned,'predicted_probability':p,'predicted_churn':(p>=threshold).astype(int),
                  'actual_viewing_hours':t.watch_hours,'predicted_viewing_hours':rp}).to_csv(ROOT/'results/test_predictions.csv',index=False)
    # Common study cases drawn from validation, never selected by test outcomes.
    cases=[]
    for label,q in [('A',.1),('B',.5),('C',.9)]:
        ix=int(np.argmin(abs(vp-np.quantile(vp,q)))); row=b.iloc[ix]
        cases.append({'scenario_id':label,**{c:float(row[c]) if c in NUM else row[c] for c in CHURN_FEATURES},'probability':float(vp[ix])})
    (ROOT/'study/scenarios.json').write_text(json.dumps(cases,indent=2))
    d[CHURN_FEATURES].head(5).to_csv(ROOT/'data/batch_example.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    teal='#087f8c'; blue='#2357a5'; red='#c15d42'
    fig,ax=plt.subplots(1,2,figsize=(11,3.7))
    fpr,tpr,_=roc_curve(t.churned,p);ax[0].plot(fpr,tpr,color=teal,lw=2,label=f"AUC = {metrics['churn_test']['roc_auc']:.3f}")
    ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Churn discrimination');ax[0].legend()
    frac,mean=calibration_curve(t.churned,p,n_bins=8,strategy='quantile');ax[1].plot(mean,frac,'o-',color=blue);ax[1].plot([0,1],[0,1],'--',color='gray');ax[1].set(xlabel='Mean predicted probability',ylabel='Observed churn fraction',title='Probability calibration')
    fig.tight_layout();fig.savefig(ROOT/'results/churn_validation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.7))
    ax[0].scatter(t.watch_hours,rp,s=13,alpha=.4,color=teal);ax[0].plot([0,110],[0,110],'--',color='gray');ax[0].set(xlabel='Actual recorded hours',ylabel='Predicted hours',title='Linear regression: held-out customers')
    ax[1].bar(['Linear model','Mean baseline'],[stats['rmse'],baseline_reg['rmse']],color=[teal,red]);ax[1].set(ylabel='RMSE in hours (lower is better)',title='Does the model add value?')
    fig.tight_layout();fig.savefig(ROOT/'results/linear_validation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.7))
    for col,axis,title in [('last_login_days',ax[0],'Churn by login recency'),('watch_hours',ax[1],'Churn by recorded viewing')]:
        bins=[-1,15,30,45,60] if col=='last_login_days' else [-1,5,10,20,111]
        group=d.groupby(pd.cut(d[col],bins),observed=True).churned.agg(['mean','size'])
        axis.bar([str(x) for x in group.index],100*group['mean'],color=teal);axis.set(title=title,ylabel='Churn %');axis.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(ROOT/'results/customer_patterns.png',dpi=180);plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ['chosen_C','threshold','churn_test','linear_test','linear_baseline','linear_interval_radius']},indent=2))

if __name__=='__main__':
    main()
