# Verification record

Date: 27 September 2026.

All six automated checks passed using Python 3.12.14, scikit-learn 1.8.0 and Streamlit 1.50.0:

1. Unique customer split membership; 3,000 training / 1,000 validation / 1,000 test; test prediction accuracy reconciles with reported metrics.
2. Excluded circular/outcome fields; numeric scaling parameters match training customers only.
3. Feature contributions plus intercept reconstruct the displayed churn probability.
4. Valid input acceptance and rejection of fractional profile counts and unknown categories.
5. Response deduplication, conflicting-ID rejection, unique theme counts and exact-quotation validation.
6. Streamlit page rendering, individual prediction, incomplete-consent blocking and the complete survey-to-download workflow.

Test responses exist only as transient test fixtures marked TEST ONLY. They are not actual participant feedback and are not saved in study outputs.

The PDF was rendered and inspected for layout. The public Streamlit URL has not been created or tested. End-to-end live deployment, real browser upload/download and receipt of actual participant responses must still be verified after publishing.

Re-run after installation from the project root:

```bash
python verify.py
```
