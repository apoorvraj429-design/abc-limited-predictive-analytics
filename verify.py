"""Meaningful reproducibility and application checks. Test answers never become study data."""
import json
import unittest
import numpy as np
import pandas as pd
from core import ROOT, CHURN_FEATURES, LINEAR_FEATURES, load_artifacts, validate_inputs, explanations
from study_tools import COLUMNS, RATINGS, OPEN, merge_responses, coding_template, theme_summary

class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log,cls.lin,cls.meta=load_artifacts()
        cls.d=pd.read_csv(ROOT/'data/abc_customers.csv')

    def test_split_and_test_metrics(self):
        splits=pd.read_csv(ROOT/'results/split_membership.csv')
        self.assertTrue(splits.customer_id.is_unique)
        self.assertEqual(splits.split.value_counts().to_dict(),{'train':3000,'validation':1000,'test':1000})
        test=pd.read_csv(ROOT/'results/test_predictions.csv')
        self.assertEqual(set(test.customer_id),set(splits.loc[splits.split=='test','customer_id']))
        self.assertAlmostEqual((test.actual_churn==test.predicted_churn).mean(),self.meta['churn_test']['accuracy'])
        self.assertEqual(len(test),1000)

    def test_no_circular_features_and_scaler_training_only(self):
        for name in ['churned','customer_id','monthly_fee','avg_watch_time_per_day','age','gender']:
            self.assertNotIn(name,CHURN_FEATURES)
        self.assertNotIn('watch_hours',LINEAR_FEATURES)
        splits=pd.read_csv(ROOT/'results/split_membership.csv')
        train=self.d[self.d.customer_id.isin(splits.loc[splits.split=='train','customer_id'])]
        np.testing.assert_allclose(self.log.named_steps['prep'].named_transformers_['num'].mean_,train[['watch_hours','last_login_days','number_of_profiles']].mean())

    def test_explanations_reconstruct_prediction(self):
        row=self.d[CHURN_FEATURES].iloc[[0]]
        e=explanations(self.log,row)
        total=e['Contribution to log-odds'].sum()+self.log.named_steps['model'].intercept_[0]
        self.assertAlmostEqual(1/(1+np.exp(-total)),self.log.predict_proba(row)[0,1],places=10)

    def test_input_validation(self):
        row=self.d[CHURN_FEATURES].iloc[[0]].copy()
        validate_inputs(row,self.meta['schema'])
        row['number_of_profiles']=2.5
        with self.assertRaises(ValueError):validate_inputs(row,self.meta['schema'])
        row['number_of_profiles']=2;row['device']='UNSEEN'
        with self.assertRaises(ValueError):validate_inputs(row,self.meta['schema'])

    def test_survey_dedup_quotes_and_counts(self):
        r={c:'test only' for c in COLUMNS};r.update({'respondent_id':'TEST-NOT-A-PARTICIPANT','consent':True})
        for c in RATINGS:r[c]=3
        for s in ['A','B','C']:
            r[s+'_before_priority']='Low';r[s+'_after_priority']='High'
        d=pd.DataFrame([r]);actual=merge_responses([d,d])
        self.assertEqual(len(actual),1)
        coded=coding_template(actual);coded.loc[0,'themes_semicolon_separated']='TRUST';coded.loc[0,'exact_quote']='test only'
        self.assertEqual(int(theme_summary(coded,actual).respondents_mentioning.iloc[0]),1)
        coded.loc[0,'exact_quote']='invented quotation'
        with self.assertRaises(ValueError):theme_summary(coded,actual)
        conflict=d.copy();conflict['role']='changed'
        with self.assertRaises(ValueError):merge_responses([d,conflict])

    def test_streamlit_pages_and_full_survey(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        for page in ['Customer check','Batch review','Model evidence','Study analysis']:
            app.sidebar.radio[0].set_value(page).run();self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Customer check').run();app.button[0].click().run()
        self.assertFalse(app.exception);self.assertGreater(len(app.metric),2)
        app.sidebar.radio[0].set_value('Manager study').run();app.button[0].click().run()
        self.assertGreater(len(app.error),0)  # consent/incomplete form must block
        app.checkbox[0].check();app.selectbox[0].set_value('Manager');app.text_input[0].set_value('TEST ONLY - prioritisation')
        for r in app.radio:
            if r.label.startswith('Your retention-review'):r.set_value('Medium')
        app.button[0].click().run();self.assertFalse(app.exception)
        self.assertIn('study_before',app.session_state)
        for r in app.radio:
            if r.label.startswith('Your final priority'):r.set_value('High')
            elif r.key and r.key.startswith('rate_'):r.set_value(3)
        for q in app.text_area:q.set_value('TEST ONLY - not a real participant response.')
        app.button[0].click().run();self.assertFalse(app.exception)
        self.assertIn('study_complete',app.session_state)
        self.assertEqual(len(merge_responses([app.session_state.study_complete])),1)

if __name__=='__main__':unittest.main(verbosity=2)
