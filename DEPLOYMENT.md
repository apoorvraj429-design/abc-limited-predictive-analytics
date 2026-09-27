# Publish ABC Limited

## Recommended: GitHub and Streamlit Community Cloud

1. Extract the project ZIP on your computer. Open the `abc_limited_project` folder.
2. In your GitHub account, create a repository named `abc-limited-predictive-analytics`. A private repository is suitable if you do not want the dataset or source publicly downloadable. Give Streamlit access to that repository when prompted.
3. Choose **Add file > Upload files** and upload the contents of the project folder. Preserve `data`, `models`, `results` and `study` as folders. `app.py` and `requirements.txt` should be in the repository root. Upload extracted files, not the ZIP. The optional hidden `.streamlit` folder supplies the colour theme; the app also works without it.
4. Open https://share.streamlit.io/ and sign in with the GitHub account that owns the repository.
5. Choose **Create app**, then **Yup, I have an app** if prompted.
6. Select your repository and branch (usually `main`). Set **Main file path** to `app.py`.
7. In **Advanced settings**, choose **Python 3.12**. No secrets or API keys are required.
8. Click **Deploy**. When it finishes, copy the actual `https://...streamlit.app` URL. That is the assignment's app link.
9. Open the link in a private/incognito window and verify that participants can access it. If the app is restricted, use Streamlit's sharing controls to grant intended viewers access or make the app public as appropriate.
10. Test Customer check, download a batch example and complete a test survey. Delete any test response from the real study folder.

## Before inviting respondents

- Tell participants to choose **Manager study first**, before other pages.
- Ask them to download the final CSV and return it through your agreed private channel. Responses are not centrally saved.
- Give each participant the same task and avoid telling them what answers you hope to obtain.
- Do not upload actual feedback to your public source repository.

## Troubleshooting

- **No module named core/study_tools:** upload the corresponding `.py` files next to `app.py`.
- **Model file not found:** verify `models/churn.joblib` and `models/viewing.joblib` are present.
- **Metrics/image/scenario not found:** upload the complete `results` and `study` folders.
- **Dependency or model-version error:** use the supplied requirements unchanged and Python 3.12. Keep scikit-learn at 1.8.0 because that is the training version.
- **Input CSV rejected:** use the downloadable batch template. Unknown categories, missing values, fractional profile counts and values outside the training range are rejected with explanations.
- **Lost feedback:** an undownloaded survey cannot be recovered after its session ends. Ask participants to download and return the CSV before closing the page.

## Local backup for the presentation

Install dependencies beforehand, then run `python -m streamlit run app.py` from the project folder. Keep the report and results available in case the venue internet is unreliable. A localhost address is not a public submission URL.

Official references checked 27 September 2026:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account
