# ABC Limited: customer decisions and managerial AI adoption

Academic project prepared for Apoorv Raj, PRM46, Roll No. P46011.

## What is complete

- Audited 5,000 supplied customer records and preserved the original values under an ABC filename.
- Trained logistic regression for the recorded churn label and ordinary linear regression for recorded viewing hours.
- Evaluated on 1,000 held-out customers, after a 3,000/1,000/1,000 training/validation/test split.
- Built a Streamlit app with individual predictions, explanations, batch review, model evidence, participant survey and researcher analysis.
- Prepared a questionnaire, consistent scenarios, blank response template, qualitative codebook and analysis workflow.
- Included trained models, reproducible code, test predictions, metrics, charts and a project report.

## What still needs to happen

1. Publish the app to your GitHub/Streamlit account and record its actual live URL.
2. Have actual managers/professionals use the study and return their downloaded responses.
3. Analyse those responses, complete qualitative coding and add the actual findings to the submission.

No participant data, quotes, completed interviews, deployment URL or adoption findings have been invented.

## Dataset and model boundaries

The matching Kaggle source describes synthetic customer behaviour, not genuine confidential industry records. The selected dataset is retained as requested. If the professor requires genuine observations even for Kaggle datasets, obtain acceptance or replace the data before claiming full compliance.

The file has no timestamps, churn horizon, measurement dates or currency definition. The tool predicts the snapshot churn label; it does not claim next-month churn. Viewing hours represent an unspecified observation window. Do not label them monthly hours or a future forecast.

The daily-watch field is derived from watch_hours/(last_login_days+1), rounded, and includes 10 values above 24. It is excluded from both models. Monthly fee is fixed by plan; only plan is retained. Age, gender and customer ID are not predictive inputs. For linear regression, churn and watch_hours are also excluded from inputs to avoid using the outcome or circular information.

## Results

At the validation-selected threshold of 0.61, held-out churn accuracy is 88.5%, precision 91.3%, recall 85.3%, F1 0.882 and ROC-AUC 0.967. The constant baseline has 50.3% accuracy and 0.500 ROC-AUC.

Linear regression RMSE is 12.392 hours versus 12.386 for the training-mean baseline. R² is -0.001. It has not demonstrated useful predictive value and is visibly marked as unsuitable for business decisions. The model remains included to satisfy the modelling exercise transparently.

## Run locally

Use Python 3.12. From this project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses the saved models; it does not retrain on startup. It has no API-key requirement and sends no automated messages. An internet connection is required for installing dependencies and publishing; the app can run locally after installation.

## Reproduce the models

```bash
python train.py
```

This regenerates saved models and results from `data/abc_customers.csv`. It fits preprocessing only on training customers. The deployed models are the evaluated training-fitted models, not a later full-data refit. Results use fixed seed 42 and pinned packages. Never load untrusted joblib model files.

## Collect and analyse actual feedback

Follow `study/PROTOCOL.md`. Participants begin at Manager study before exploring the rest of the app. They download one response CSV and return it to the researcher. The app has no central response database. A completed screen is not proof that the researcher received a response.

Upload returned files in Study analysis, or run:

```bash
python study_tools.py /path/to/response1.csv /path/to/response2.csv --out private_study_results
python study_tools.py /path/to/response1.csv /path/to/response2.csv --coded /path/to/completed_coding.csv --out private_study_results
```

Do not publish collected responses in the GitHub repository. Keep them in a private folder. The app analyses uploaded data in the current session and requires downloading the results to retain them.

## Deliverables

- `app.py`, `core.py`, `study_tools.py`: deployed application and shared logic.
- `train.py`, `requirements.txt`: reproducible training and dependencies.
- `models/`: the two fitted pipelines.
- `results/`: audit, performance, coefficients, split membership, held-out predictions and figures.
- `study/`: survey protocol, scenarios, blank response CSV, codebook and findings template.
- `DEPLOYMENT.md`: GitHub and Streamlit instructions.
- `SOURCE.md`: source attribution and evidence boundaries.

The detailed teaching explanation and presentation deck are deliberately deferred to the next requested stages.
