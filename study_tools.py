"""Collect and summarise actual responses. No responses are generated here."""
import argparse
from pathlib import Path
import pandas as pd
from core import ROOT, safe_csv

RATINGS = {
 'usefulness':'This tool would help me make customer-retention decisions.',
 'ease_of_use':'I found the tool easy to understand and use.',
 'trust':'I trust the churn score enough to consider it alongside other evidence.',
 'explanation_value':'The explanation helped me understand the churn score.',
 'adoption_intent':'I would consider using a validated version of this tool at work.',
 'wrong_decision_concern':'Concern about making a wrong decision would discourage my use.'}
OPEN = {
 'useful_tasks':'For which tasks would this tool be useful, and where would it not help?',
 'likes':'What did you like about the tool?',
 'dislikes':'What did you dislike or find confusing?',
 'trust_reasons':'Why did you trust or distrust the prediction? What evidence would increase your trust?',
 'experience_conflict':'If your experience and the model disagree, what would you do, and why?',
 'explainability':'Did seeing the explanation affect your willingness to use the tool? How?',
 'adoption_barriers':'What personal or organisational factors would discourage adoption?',
 'linear_reaction':'The viewing-hours model performed worse than a simple average. How should a manager respond?',
 'improvements':'What would need to change before you would use this tool at work?'}
THEMES = {
 'USEFULNESS':'Time saving, prioritisation or decision relevance.',
 'TRUST':'Reliability, validation, data quality or uncertainty.',
 'EXPLAINABILITY':'Understanding drivers or wanting clearer explanations.',
 'HUMAN_JUDGMENT':'Experience, overrides, verification or shared decisions.',
 'ERROR_ACCOUNTABILITY':'Fear of mistakes, responsibility or consequences.',
 'ORGANISATION':'Leadership, cost, integration, training or workflow barriers.',
 'USABILITY':'Interface, wording, effort or accessibility.',
 'DATA_LIMITS':'Synthetic data, missing context, timing or weak regression.',
 'OTHER':'An emergent theme; explain it in the coding note.'}
BASE = ['respondent_id','consent','role','years_experience','decision_context','submitted_utc']
CASE_FIELDS = [f'{s}_{v}' for s in ['A','B','C'] for v in ['before_priority','after_priority','reason']]
COLUMNS = BASE + CASE_FIELDS + list(RATINGS) + list(OPEN)

def merge_responses(frames):
    combined=pd.concat(frames,ignore_index=True).fillna('')
    missing=set(COLUMNS)-set(combined.columns)
    if missing: raise ValueError('Response file is missing columns: '+', '.join(sorted(missing)))
    if combined.empty: return combined[COLUMNS]
    combined=combined[COLUMNS].drop_duplicates()
    combined['respondent_id']=combined.respondent_id.astype(str).str.strip()
    if (combined.respondent_id=='').any(): raise ValueError('Every response needs a respondent ID.')
    if combined.respondent_id.duplicated().any(): raise ValueError('Conflicting responses have the same respondent ID. Resolve them before analysis.')
    if not combined.consent.astype(str).str.lower().isin(['true','yes','1']).all():
        raise ValueError('Only responses with recorded consent can be analysed.')
    for col in RATINGS:
        v=pd.to_numeric(combined[col],errors='coerce')
        if v.isna().any() or not (v.between(1,5)&(v==v.round())).all(): raise ValueError(f'{col} must be an integer from 1 to 5.')
        combined[col]=v.astype(int)
    for col in [c for c in CASE_FIELDS if not c.endswith('reason')]:
        if not combined[col].isin(['Low','Medium','High']).all(): raise ValueError(f'Invalid priority in {col}.')
    return combined.reset_index(drop=True)

def rating_summary(d):
    if d.empty:return pd.DataFrame(columns=['measure','n','mean','median','rating_4_or_5_pct'])
    return pd.DataFrame([{'measure':c,'n':len(d),'mean':d[c].mean(),'median':d[c].median(),
      'rating_4_or_5_pct':100*d[c].ge(4).mean(), **{f'rating_{i}_count':int(d[c].eq(i).sum()) for i in range(1,6)}} for c in RATINGS])

def decision_summary(d):
    return pd.DataFrame([{'scenario':s,'n':len(d),'changed_decision_count':int((d[f'{s}_before_priority']!=d[f'{s}_after_priority']).sum()),
       'changed_decision_pct':float(100*(d[f'{s}_before_priority']!=d[f'{s}_after_priority']).mean()) if len(d) else 0} for s in ['A','B','C']])

def coding_template(d):
    records=[]
    for _,r in d.iterrows():
        for q in list(OPEN)+[f'{s}_reason' for s in ['A','B','C']]:
            if str(r[q]).strip(): records.append({'respondent_id':r.respondent_id,'question':q,'response':r[q],
                 'themes_semicolon_separated':'','exact_quote':'','interpretation':'','coder':''})
    return pd.DataFrame(records,columns=['respondent_id','question','response','themes_semicolon_separated','exact_quote','interpretation','coder'])

def theme_summary(coded, responses):
    needed={'respondent_id','question','response','themes_semicolon_separated','exact_quote','interpretation','coder'}
    if not needed.issubset(coded.columns): raise ValueError('Use the downloaded qualitative coding template.')
    truth=coding_template(responses).set_index(['respondent_id','question']).response.to_dict()
    seen=set(); out=[]
    for _,r in coded.fillna('').iterrows():
        key=(str(r.respondent_id),str(r.question))
        if key not in truth or str(r.response)!=str(truth[key]): raise ValueError('Coding response does not match the original response.')
        if key in seen: raise ValueError('Duplicate respondent/question coding row.')
        seen.add(key)
        if str(r.exact_quote) and str(r.exact_quote) not in str(r.response): raise ValueError('Quotes must be copied exactly from the response.')
        codes=[c.strip() for c in str(r.themes_semicolon_separated).split(';') if c.strip()]
        for code in codes:
            if code not in THEMES:raise ValueError('Unknown theme: '+code)
            out.append({'respondent_id':r.respondent_id,'theme':code})
    if not out:return pd.DataFrame(columns=['theme','respondents_mentioning','share_of_sample_pct'])
    counts=pd.DataFrame(out).drop_duplicates().groupby('theme').size().rename('respondents_mentioning').reset_index()
    counts['share_of_sample_pct']=100*counts.respondents_mentioning/len(responses)
    return counts.sort_values('respondents_mentioning',ascending=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('responses',nargs='+');parser.add_argument('--coded');parser.add_argument('--out',default='private_study_results')
    args=parser.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    d=merge_responses([pd.read_csv(p,keep_default_na=False,dtype={'respondent_id':str}) for p in args.responses])
    for name,frame in [('responses',d),('rating_summary',rating_summary(d)),('decision_summary',decision_summary(d)),('qualitative_coding',coding_template(d))]:
        (out/f'{name}.csv').write_bytes(safe_csv(frame))
    if args.coded:
        summary=theme_summary(pd.read_csv(args.coded,keep_default_na=False,dtype={'respondent_id':str}),d)
        (out/'theme_summary.csv').write_bytes(safe_csv(summary))
    print(f'Analysed {len(d)} actual respondents. Qualitative interpretation requires human coding; no findings are invented.')

if __name__=='__main__':main()
