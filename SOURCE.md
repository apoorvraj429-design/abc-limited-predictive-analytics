# Data provenance and references

The case name and app brand are ABC Limited. That is a pseudonym for the academic case, not a claim that an actual company supplied the file.

## Uploaded data

Source of bytes: the CSV supplied by the user in this conversation. The copy in this package is `data/abc_customers.csv`, with unchanged data values.

SHA-256: `16e00950956f12921625001315dff2c450c9ca14af885cb19b4035b0b6fb311e`.

Rows: 5,000. Columns: 14. No nulls or duplicate customer IDs. The matching public dataset listing is by Abdul Wadood on Kaggle:

https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset

The publisher's indexed description calls this synthetic customer-behaviour data. The listing matches the filename, schema and size; an independent byte-for-byte download comparison was not performed. It is therefore a matched source, not a verified chain of custody. The original brand appears only in the source URL for traceability. No real-company affiliation or endorsement is claimed. Dataset licensing was not independently verified; the source attribution does not create a new licence for the dataset.

## Sources used for implementation

- scikit-learn, LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn, LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- scikit-learn, leakage and preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
- Streamlit, application testing: https://docs.streamlit.io/develop/api-reference/app-testing
- Streamlit, deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit, dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

Access/review date: 27 September 2026. Model metrics and data-quality findings are computed directly from the supplied CSV, not copied from external notebooks.
