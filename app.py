"""ABC Limited | Predictive Analytics & Managerial AI Adoption."""
import json
import uuid
import base64
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from core import ROOT, NUM, CAT, CHURN_FEATURES, LINEAR_FEATURES, LABELS, load_artifacts, validate_inputs, explanations, safe_csv
from study_tools import RATINGS, OPEN, THEMES, COLUMNS, merge_responses, rating_summary, decision_summary, coding_template, theme_summary

st.set_page_config(page_title='ABC Limited | Customer decisions',page_icon='◉',layout='wide')
st.markdown('''<style>
.block-container{padding-top:3.5rem;max-width:1200px}h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:white;border:1px solid #dce7ed;border-radius:14px;padding:18px}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] *,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] *{color:#193549!important}
[data-testid="stMetric"] [data-testid="stMetricLabel"]{font-weight:600}
div[data-testid="stSidebarContent"]{padding-top:1rem}.stButton>button{border-radius:8px}
.abc-hero{display:flex;align-items:center;gap:24px;background:linear-gradient(120deg,#102e46,#125765);border:1px solid #2d6975;border-radius:22px;padding:30px 34px;margin:8px 0 26px;color:#f4fbff}
.abc-hero-copy{flex:1;min-width:0}.abc-hero h1{font-size:clamp(1.8rem,3.2vw,2.65rem);line-height:1.12;margin:10px 0 16px;color:#fff;letter-spacing:-.035em}
.abc-hero p{color:#d3e9ee;font-size:1.05rem;line-height:1.55;margin:0}.abc-eyebrow{color:#77dccb;font-size:.76rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}
.abc-hero img{width:34%;max-width:330px;min-width:180px}.abc-pill{display:inline-block;background:#ffffff12;border:1px solid #ffffff30;border-radius:30px;color:#c8eee7;font-size:.76rem;padding:5px 11px;margin:18px 7px 0 0}
.abc-brand{display:flex;align-items:center;gap:12px;margin-bottom:8px}.abc-brand-mark{background:#087f8c;color:white;border-radius:12px;padding:12px 9px;font-size:18px;font-weight:800;letter-spacing:-1px}.abc-brand-name{font-size:22px;font-weight:750}
.abc-panel{border:1px solid #7b9da440;border-radius:18px;padding:22px;margin:8px 0 20px;background:var(--secondary-background-color,transparent)}
.abc-panel h3{font-size:1.18rem;margin:0 0 4px}.abc-panel p{font-size:.88rem;opacity:.8;margin:0 0 12px}
.abc-composition{display:flex;align-items:center;justify-content:center;gap:26px;flex-wrap:wrap}.abc-composition img{width:165px}.abc-legend-row{display:flex;align-items:center;gap:10px;padding:9px 0}.abc-dot{width:12px;height:12px;border-radius:50%;display:inline-block}.abc-legend-row strong{font-size:1.15rem}.abc-legend-row span{font-size:.9rem}
@media(max-width:640px){.abc-hero{padding:24px;flex-direction:column;align-items:flex-start}.abc-hero img{width:100%;max-width:260px;align-self:center}.abc-composition{gap:12px}}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def artifacts():return load_artifacts()
log,lin,meta=artifacts();schema=meta['schema'];threshold=meta['threshold']

def svg_uri(svg):
    return 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode('utf-8')).decode('ascii')

HERO_SVG='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 250">
<circle cx="225" cy="120" r="109" fill="#ffffff08"/><circle cx="225" cy="120" r="86" fill="none" stroke="#ffffff14"/>
<rect x="86" y="55" width="246" height="145" rx="15" fill="#eaf7fb"/><rect x="101" y="70" width="216" height="109" rx="7" fill="#16495c"/>
<rect x="173" y="201" width="69" height="7" rx="3" fill="#a3c9d2"/><rect x="188" y="208" width="40" height="14" fill="#a3c9d2"/><rect x="156" y="222" width="105" height="6" rx="3" fill="#eaf7fb"/>
<rect x="113" y="82" width="121" height="84" rx="6" fill="#21697a"/><circle cx="174" cy="123" r="26" fill="#7bdbc4"/><path d="M167 109L187 123L167 137Z" fill="#123f50"/>
<rect x="246" y="84" width="58" height="11" rx="5" fill="#6ed8c2"/><rect x="246" y="105" width="42" height="6" rx="3" fill="#75a9b5"/><rect x="246" y="120" width="53" height="6" rx="3" fill="#75a9b5"/><rect x="246" y="145" width="47" height="18" rx="8" fill="#367e8c"/>
<rect x="20" y="148" width="114" height="77" rx="13" fill="#b9eedf"/><circle cx="49" cy="177" r="10" fill="#198c82"/><path d="M33 201Q33 184 49 184Q65 184 65 201" fill="#198c82"/><rect x="76" y="170" width="43" height="7" rx="3" fill="#238f87"/><rect x="76" y="186" width="30" height="6" rx="3" fill="#58aaa1"/>
<rect x="276" y="21" width="101" height="58" rx="13" fill="#c4ddff"/><circle cx="301" cy="49" r="13" fill="#366eb2"/><path d="M295 49L299 53L307 44" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/><rect x="322" y="38" width="39" height="6" rx="3" fill="#4076b4"/><rect x="322" y="51" width="27" height="5" rx="2" fill="#77a0cf"/>
<circle cx="57" cy="60" r="17" fill="#f0b87b"/><path d="M50 60H64M57 53V67" stroke="#744e27" stroke-width="3" stroke-linecap="round"/>
</svg>'''

def overview_graphic():
    n=meta['data_audit']['rows'];churn=meta['data_audit']['churn_count'];retained=n-churn;share=100*churn/n
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180"><circle cx="90" cy="90" r="70" fill="none" stroke="#36b8a0" stroke-width="22"/><circle cx="90" cy="90" r="70" fill="none" stroke="#eea46b" stroke-width="22" pathLength="100" stroke-dasharray="{share} {100-share}" transform="rotate(-90 90 90)"/><circle cx="90" cy="90" r="53" fill="#eef8f8"/><text x="90" y="88" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#193549">{n:,}</text><text x="90" y="108" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#456675">CUSTOMERS</text></svg>'''
    st.markdown(f'''<div class="abc-panel"><h3>Customer snapshot</h3><p>Recorded outcomes in the supplied dataset</p><div class="abc-composition"><img src="{svg_uri(svg)}" alt="{churn:,} churned customers and {retained:,} retained customers"/><div><div class="abc-legend-row"><i class="abc-dot" style="background:#eea46b"></i><div><strong>{churn:,}</strong><br/><span>Churned · {share:.1f}%</span></div></div><div class="abc-legend-row"><i class="abc-dot" style="background:#36b8a0"></i><div><strong>{retained:,}</strong><br/><span>Retained · {100-share:.1f}%</span></div></div></div></div></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="abc-brand"><div class="abc-brand-mark">ABC</div><div class="abc-brand-name">ABC Limited</div></div>',unsafe_allow_html=True)
    st.caption('CUSTOMER DECISION LAB')
    page=st.radio('Workspace',['Overview','Customer check','Batch review','Model evidence','Manager study','Study analysis'],label_visibility='collapsed')
    st.divider()
    st.caption('Academic prototype • visual update 1.1')
    st.caption('Synthetic data. Human review required. No verified future prediction period.')

def risk(p):return 'Higher priority' if p>=threshold else 'Lower priority'
def metrics_row(p):
    a,b,c=st.columns(3);a.metric('Churn score',f'{p:.1%}');b.metric('Review priority',risk(p));c.metric('Review threshold',f'{threshold:.0%}')
    st.progress(float(p),text=f'Model churn score: {p:.1%} | Review threshold: {threshold:.0%}')
def show_explanation(row, key):
    exp=explanations(log,pd.DataFrame([row]))
    st.bar_chart(exp.set_index('Factor')['Contribution to log-odds'],horizontal=True,color='#087f8c')
    st.caption('Positive values raise the model score; negative values lower it. These are exact model contributions in log-odds, not percentage-point changes or causal effects. Comparison: training averages and reference categories.')
    with st.expander('View the exact contribution values'):
        st.dataframe(exp,hide_index=True,width="stretch")

if page=='Overview':
    st.markdown(f'''<div class="abc-hero"><div class="abc-hero-copy"><div class="abc-eyebrow">ABC Limited · Customer Decision Lab</div><h1>Understand customers.<br/>Make informed decisions.</h1><p>Explore customer churn, inspect the evidence and understand when managers choose to use AI advice.</p><span class="abc-pill">Predictive analytics</span><span class="abc-pill">Human judgment</span><span class="abc-pill">Synthetic-data prototype</span></div><img src="{svg_uri(HERO_SVG)}" alt="Illustration of a streaming screen, customer profile and review check"/></div>''',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric('Customer records','5,000');b.metric('Recorded churn rate',f"{meta['data_audit']['churn_rate']:.1%}");c.metric('Held-out test customers','1,000')
    st.info('Taking part in the research? Open Manager study first, before exploring customer predictions.')
    overview_graphic()
    st.markdown('### Two models, two different conclusions')
    a,b=st.columns(2)
    with a:
        st.markdown('#### ◎ Churn review')
        st.write('Logistic regression estimates the probability of the recorded churn label. Use it to practise prioritisation and inspect the factors behind the score.')
        st.metric('Test ROC-AUC',f"{meta['churn_test']['roc_auc']:.3f}")
    with b:
        st.markdown('#### ◷ Viewing-hours estimate')
        st.write('Linear regression estimates recorded viewing hours from the other available customer attributes. It does not outperform a simple average on the test set.')
        st.metric('Test R²',f"{meta['linear_test']['r2']:.3f}")
    st.info('Start with Customer check. For the research study, participants should begin directly at Manager study, before exploring other pages.')
    st.image(str(ROOT/'results/customer_patterns.png'),caption='Descriptive associations in the supplied synthetic dataset; not causal effects.')
    with st.expander('Data quality and responsible interpretation'):
        st.write('The matching Kaggle data card describes synthetic customer behaviour. No actual company affiliation or production performance is claimed. There are no timestamps or documented pre-churn measurement windows.')
        st.write('The daily viewing column is derived from recorded viewing hours divided by days since login plus one. It includes 10 values above 24 and is excluded. The fee is fixed by subscription plan and is excluded as a duplicate input. IDs, age and gender are also excluded from prediction.')

elif page=='Customer check':
    st.title('Review a customer')
    st.write('Enter recorded behaviour and subscription details. The score supports review, not automatic action.')
    with st.form('customer_form'):
        cols=st.columns(3);row={}
        for i,c in enumerate(NUM):
            with cols[i%3]:
                lo,hi=schema['ranges'][c]
                row[c]=st.number_input(LABELS[c],min_value=float(lo),max_value=float(hi),value=float(schema['defaults'][c]),step=.1 if c=='watch_hours' else 1.)
        for i,c in enumerate(CAT):
            with cols[i%3]:row[c]=st.selectbox(LABELS[c],schema['categories'][c])
        submitted=st.form_submit_button('Review customer',type='primary')
    if submitted:
        try:
            x=validate_inputs(pd.DataFrame([row]),schema)
            st.session_state.customer_result=x.to_dict('records')[0]
        except ValueError as e:st.error(str(e))
    if 'customer_result' in st.session_state:
        row=st.session_state.customer_result;x=pd.DataFrame([row]);p=float(log.predict_proba(x)[:,1][0])
        st.markdown('### Churn review');metrics_row(p)
        st.write('Suggested next step: '+('Review recent customer activity and seek context before deciding whether outreach is appropriate.' if p>=threshold else 'Continue routine monitoring; a lower score does not rule out churn.'))
        show_explanation(row,'single')
        st.markdown('### Viewing-hours estimate')
        val=float(lin.predict(x[LINEAR_FEATURES])[0]);radius=meta['linear_interval_radius']
        a,b=st.columns(2);a.metric('Estimated recorded hours',f'{max(0,val):.1f} h');b.metric('90% reference interval',f'{max(0,val-radius):.1f}–{max(0,val+radius):.1f} h')
        st.warning('This linear model performs worse than a simple training-average baseline. Do not use this estimate for customer actions or business planning. It estimates the dataset’s unspecified observation window, not next month.')
        st.caption('Interval calibrated on validation residuals under exchangeability. It is not a guarantee for real customers. Viewing-hours input above is used only by the churn model.')
        result=pd.DataFrame([{**row,'churn_score':p,'review_threshold':threshold,'review_priority':risk(p),'linear_raw_hours':val,'linear_status':'Not recommended for decisions'}])
        st.download_button('Download this review',safe_csv(result),'abc_customer_review.csv','text/csv')

elif page=='Batch review':
    st.title('Review a customer list')
    st.write('Upload up to 10,000 rows using the template. No customer names, email addresses or payment details are needed.')
    st.download_button('Download example input CSV',(ROOT/'data/batch_example.csv').read_bytes(),'abc_batch_example.csv','text/csv')
    upload=st.file_uploader('Customer CSV',type=['csv'],key='batch')
    if upload:
        try:
            original=pd.read_csv(upload);x=validate_inputs(original,schema);p=log.predict_proba(x)[:,1]
            out=x.copy();out.insert(0,'input_row',np.arange(1,len(out)+1));out['churn_score']=p
            out['review_priority']=np.where(p>=threshold,'Higher priority','Lower priority')
            out=out.sort_values('churn_score',ascending=False)
            a,b=st.columns(2);a.metric('Customers reviewed',len(out));b.metric('Above review threshold',int((p>=threshold).sum()))
            st.dataframe(out,hide_index=True,width="stretch")
            st.download_button('Download ranked review list',safe_csv(out),'abc_batch_review.csv','text/csv')
            st.caption('Ranking is an academic demonstration. The example rows come from the modelling dataset and are not independent validation evidence.')
        except (ValueError,pd.errors.ParserError) as e:st.error(str(e))

elif page=='Model evidence':
    st.title('Evidence before adoption')
    st.write('3,000 training customers • 1,000 validation customers • 1,000 untouched test customers. Fixed random seed: 42. The split is stratified by churn.')
    st.markdown('### Logistic regression')
    m=meta['churn_test'];base=meta['churn_baseline']
    comparison=pd.DataFrame({'Measure':['Accuracy','Precision','Recall','F1','ROC-AUC','Brier score'],
        'Logistic regression':[m[k] for k in ['accuracy','precision','recall','f1','roc_auc','brier']],
        'Constant baseline':[base[k] for k in ['accuracy','precision','recall','f1','roc_auc','brier']]})
    st.dataframe(comparison.round(4),hide_index=True,width="stretch")
    st.caption(f"Regularisation C={meta['chosen_C']:g} selected by validation ROC-AUC; threshold {threshold:.2f} selected by validation F1. Brier score is better when lower. F1 tuning is not a proven financial optimum.")
    st.dataframe(pd.DataFrame(m['confusion_matrix'],index=['Actual retained','Actual churned'],columns=['Predicted retained','Predicted churned']))
    st.image(str(ROOT/'results/churn_validation.png'))
    st.markdown('### Linear regression')
    st.dataframe(pd.DataFrame({'Measure':['MAE (hours)','RMSE (hours)','R²'],
      'Linear regression':[meta['linear_test'][k] for k in ['mae','rmse','r2']],
      'Mean baseline':[meta['linear_baseline'][k] for k in ['mae','rmse','r2']]}).round(4),hide_index=True,width="stretch")
    st.image(str(ROOT/'results/linear_validation.png'))
    st.warning('No useful viewing-hours predictive gain was demonstrated. This is a model-evaluation result, not a reason to hide the model or inflate its accuracy.')
    st.markdown('### Boundaries of this evidence')
    st.write('The data is synthetic and cross-sectional. Strong churn metrics may reflect a data-generation rule. Temporal leakage cannot be ruled out without timestamps. Real deployment requires genuine pre-outcome observations, an explicit horizon, prospective validation, calibration, subgroup checks and a controlled retention experiment.')
    st.download_button('Download full model metrics',(ROOT/'results/metrics.json').read_bytes(),'abc_model_metrics.json','application/json')

elif page=='Manager study':
    st.title('How would you use AI advice?')
    st.write('A 10–15 minute academic study of managerial judgment. No right answer is expected; disagreement is useful.')
    st.info('Your response is kept only in this browser session. At the end, download the response file and return it to the researcher. It is not automatically sent or stored centrally.')
    cases=json.loads((ROOT/'study/scenarios.json').read_text())
    st.session_state.setdefault('respondent_id','M-'+uuid.uuid4().hex[:8])
    if 'study_before' not in st.session_state:
        st.markdown('### 1. Your judgment before seeing the model')
        with st.form('before'):
            consent=st.checkbox('I voluntarily agree to participate and permit anonymised responses and quotations in this academic assignment.')
            st.caption('Do not enter your name, employer, client details or other identifying information. You may stop before returning your response.')
            role=st.selectbox('Your role',['Manager','Working professional','Team leader','Supervisor','Entrepreneur','Other experienced decision-maker'],index=None)
            years=st.number_input('Years of work experience',min_value=0,max_value=60,value=2)
            context=st.text_input('What kinds of decisions do you make?')
            before={}
            for case in cases:
                s=case['scenario_id'];st.markdown(f'#### Customer {s}')
                st.write({LABELS[c]:case[c] for c in CHURN_FEATURES})
                before[f'{s}_before_priority']=st.radio(f'Your retention-review priority for customer {s}',['Low','Medium','High'],index=None,horizontal=True)
            proceed=st.form_submit_button('Save my initial judgments and show model evidence',type='primary')
        if proceed:
            if not consent or role is None or not context.strip() or any(v is None for v in before.values()):
                st.error('Please provide consent, role, decision context and a judgment for all three customers.')
            else:
                st.session_state.study_before={'respondent_id':st.session_state.respondent_id,'consent':True,'role':role,'years_experience':years,'decision_context':context,**before}
                st.rerun()
    elif 'study_complete' not in st.session_state:
        st.markdown('### 2. Review the evidence and decide again')
        st.write(f"This prototype uses synthetic records. Held-out churn ROC-AUC is {meta['churn_test']['roc_auc']:.3f} and accuracy is {meta['churn_test']['accuracy']:.1%}. No future churn period has been validated. The linear viewing model fails to beat a simple average.")
        for case in cases:
            s=case['scenario_id'];st.markdown(f'#### Customer {s}')
            st.write({LABELS[c]:case[c] for c in CHURN_FEATURES})
            st.write(f"Your initial priority: **{st.session_state.study_before[f'{s}_before_priority']}**. Model churn score: **{case['probability']:.1%}**. Model review priority: **{risk(case['probability'])}**.")
            with st.expander(f'See the explanation for customer {s}',expanded=True):show_explanation(case,s)
        with st.form('feedback'):
            responses={}
            for case in cases:
                s=case['scenario_id'];responses[f'{s}_after_priority']=st.radio(f'Your final priority for customer {s}',['Low','Medium','High'],index=None,horizontal=True)
                responses[f'{s}_reason']=st.text_area(f'Why did you keep or change your decision for customer {s}?')
            st.markdown('### 3. Your assessment')
            st.caption('1 = strongly disagree; 2 = disagree; 3 = neutral; 4 = agree; 5 = strongly agree.')
            for key,label in RATINGS.items():responses[key]=st.radio(label,[1,2,3,4,5],index=None,horizontal=True,key='rate_'+key)
            for key,label in OPEN.items():responses[key]=st.text_area(label,key='open_'+key)
            done=st.form_submit_button('Prepare my response for download',type='primary')
        if done:
            if any(v is None for v in responses.values()) or any(not str(responses[k]).strip() for k in list(OPEN)+[s+'_reason' for s in ['A','B','C']]):
                st.error('Please complete the ratings and questions. You may write “Prefer not to answer” for any open question.')
            else:
                response={**st.session_state.study_before,**responses,'submitted_utc':datetime.now(timezone.utc).isoformat()}
                st.session_state.study_complete=pd.DataFrame([response],columns=COLUMNS)
                st.rerun()
    if 'study_complete' in st.session_state:
        st.success('Your response is ready. Download it and return the file to the researcher to complete participation.')
        st.download_button('Download my response CSV',safe_csv(st.session_state.study_complete),f"{st.session_state.respondent_id}_response.csv",'text/csv',type='primary')
        st.caption('Keep your respondent ID if you later wish to ask the researcher to remove your response. Refreshing or closing the session can lose an undownloaded response.')

elif page=='Study analysis':
    st.title('Analyse actual managerial feedback')
    st.write('Researcher workspace. Upload the response CSV files returned by participants. No survey findings are pre-filled.')
    st.caption('Uploads are processed in the current session and are not written to the public repository. Download your outputs before closing the session.')
    uploads=st.file_uploader('Participant response files',type=['csv'],accept_multiple_files=True,key='responses')
    if not uploads:
        st.info('No responses loaded. Data collection and qualitative findings are pending.')
    else:
        try:
            d=merge_responses([pd.read_csv(f,keep_default_na=False,dtype={'respondent_id':str}) for f in uploads])
            st.metric('Unique consenting respondents',len(d))
            ratings=rating_summary(d);decisions=decision_summary(d);coding=coding_template(d)
            st.markdown('### Rating summaries');st.dataframe(ratings.round(2),hide_index=True,width="stretch")
            st.caption('The error-concern item runs in the opposite direction to the favourable ratings. Do not average all items into one adoption score.')
            st.markdown('### Decision changes');st.dataframe(decisions.round(2),hide_index=True,width="stretch")
            st.caption('A changed decision does not itself mean a better decision or prove that explanations caused the change.')
            st.download_button('Download combined responses',safe_csv(d),'abc_responses.csv','text/csv')
            st.download_button('Download rating summary',safe_csv(ratings),'abc_rating_summary.csv','text/csv')
            st.download_button('Download decision-change summary',safe_csv(decisions),'abc_decision_summary.csv','text/csv')
            st.markdown('### Qualitative themes')
            st.write('Read every response. Assign one or more codes, separated by semicolons, in the coding template. Include differing views and copy quotations exactly. Counts show mentions in this small sample, not population prevalence.')
            st.dataframe(pd.DataFrame(THEMES.items(),columns=['Code','Meaning']),hide_index=True,width="stretch")
            st.download_button('Download qualitative coding template',safe_csv(coding),'abc_qualitative_coding.csv','text/csv')
            coded=st.file_uploader('Upload your completed coding CSV',type=['csv'],key='coded')
            if coded:
                table=theme_summary(pd.read_csv(coded,keep_default_na=False,dtype={'respondent_id':str}),d)
                st.dataframe(table,hide_index=True,width="stretch")
                st.download_button('Download theme counts',safe_csv(table),'abc_theme_summary.csv','text/csv')
        except (ValueError,pd.errors.ParserError) as e:st.error(str(e))
