# Loan Default Risk Analytics & AI-Powered Chatbot using RAG

> An end-to-end predictive analytics solution for assessing loan default risk, complete with exploratory data analysis, machine learning models, an interactive Tableau dashboard, and a Retrieval-Augmented Generation (RAG) based chatbot assistant.  
> Developed for **BUS5PR1 – Predictive Analytics**, Semester 1, Master of Business Analytics, La Trobe University (2025).

---

## Project Purpose

Loan default prediction plays a critical role in the financial services sector. By proactively identifying high-risk applicants, banks and lenders can manage credit risk, improve lending strategies, and ensure compliance with regulatory standards.

This project aims to:
- Build accurate and explainable models for loan default prediction
- Provide visual tools to communicate insights to stakeholders
- Support non-technical users through an intelligent chatbot interface powered by **RAG + LLM**

---

## Key Objectives

✔️ Perform comprehensive Exploratory Data Analysis (EDA)  
✔️ Apply feature engineering and transformation techniques  
✔️ Train and evaluate multiple supervised ML models  
✔️ Visualize findings with a Tableau dashboard  
✔️ Build a chatbot that retrieves information from the report and returns generative answers to user queries

---

## Business Questions Answered

- What are the most important predictors of loan default?
- Can we segment borrowers into risk profiles?
- How do income, employment status, and credit history relate to default?
- How can a chatbot assist lenders and analysts in understanding the data?

---

## Project Structure

```bash
Loan-Default-Risk-Analytics-RAG-Chatbot/
├── notebooks/
│   ├── 01_loan_default_eda_and_modeling.ipynb   # Full EDA, model training and evaluation
│   └── 02_rag_chatbot_loan_support.ipynb        # LangChain RAG chatbot code
├── data/
│   ├── data_train.csv
│   ├── data_test.csv
│   └── cleaned_train_data.csv
├── reports/
│   ├── Final_report.docx                        # Project report with results & discussion
│   ├── dataset_description.pdf
│   └── project_architecture_diagram.pdf
├── dashboard/
│   └── loan_default_dashboard_tableau.twb       # Interactive Tableau file
└── README.md
```

## How to Use This Project

### nstall Python Dependencies
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost gradio langchain openai

## Run the Notebooks

- Open notebooks/01_loan_default_eda_and_modeling.ipynb for full EDA and modeling.
- Open notebooks/02_rag_chatbot_loan_support.ipynb to test the AI chatbot (requires an OpenAI API key).

---

## Chatbot Preview

This project features an interactive chatbot built using **LangChain** and **OpenAI GPT-4** that can intelligently answer queries related to:

- Model performance  
- Data features  
- Business implications  

**Example prompts:**

- "What are the top predictors of default?"
- "Summarise the EDA process."
- "Explain the difference in model accuracies."

---

## Dashboard Insights

The Tableau dashboard allows stakeholders to:

- View default rates by income, job type, and loan purpose  
- Drill into customer segments interactively  
- Understand key risk indicators visually  

📂 Open dashboard/loan_default_dashboard_tableau.twb in **Tableau Desktop** to explore.

## Model Evaluation

| Model               | AUC Score | Accuracy | Notes                            |
|--------------------|-----------|----------|----------------------------------|
| Logistic Regression| 0.982     | 95.4%    | Best model, interpretable        |
| XGBoost            | 0.953     | 91.2%    | Strong performance after tuning  |
| Random Forest      | 0.759     | 86.1%    | Some overfitting observed        |

Evaluation includes:
- Confusion matrix
- ROC-AUC curves
- SHAP explainability

## Tools & Skills Demonstrated

| Category         | Tools / Libraries                           |
|------------------|---------------------------------------------|
| Data Wrangling   | pandas, numpy                           |
| Visualisation    | matplotlib, seaborn, Tableau          |
| Machine Learning | scikit-learn, XGBoost, RandomForest   |
| AI + NLP         | LangChain, OpenAI GPT-4, Gradio       |
| Dashboarding     | Tableau Desktop                           |
| Communication    | Word, Markdown, Diagrams              |
| Version Control  | Git, GitHub                             |

## Key Files

- notebooks/01_loan_default_eda_and_modeling.ipynb  
  → ML pipeline with full EDA, feature engineering, and model training

- notebooks/02_rag_chatbot_loan_support.ipynb  
  → AI chatbot implementation using Retrieval-Augmented Generation (RAG)

- dashboard/loan_default_dashboard_tableau.twb  
  → Interactive business dashboard for risk analysis and segmentation

- reports/Final_report.docx  
  → Final written project report with insights and findings

- reports/dataset_description.pdf  
  → Dataset overview and variable descriptions

- reports/project_architecture_diagram.pdf  
  → High-level system architecture of the AI-powered analytics solution

- data/*.csv  
  → Raw and cleaned training/testing datasets used for modeling

---

## 🙋‍♂️ Author

**Phu Qui Dang**  
Master of Business Analytics (Data Science), La Trobe University  

- Portfolio Wedsite: [Portfolio Website](https://quidangthedataanalyst.framer.website/)  
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/phuquidang/)  
- GitHub: [@dangquii](https://github.com/dangquii)
