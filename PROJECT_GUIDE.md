# Project Guide — Customer Churn Prediction

## 1. Project Purpose

Build an end-to-end machine learning solution to predict customer churn for a telecommunications company.

The project aims to demonstrate practical machine learning skills, software engineering practices, model evaluation, reproducibility, and professional documentation.

## 2. Learning Objectives

- Understand supervised learning and binary classification.
- Perform exploratory data analysis (EDA).
- Prepare and validate real-world datasets.
- Build machine learning pipelines.
- Train and evaluate classification models.
- Perform error analysis and model improvement.
- Write reusable and maintainable Python code.
- Apply Git and GitHub version control.
- Write automated tests.
- Prepare a model for deployment.

## 3. Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code
- Git and GitHub
- Pytest
- FastAPI (planned extension)
- Docker (planned extension)

## 4. Development Principles

- Keep exploratory analysis separate from production-oriented code.
- Use modular and reusable Python code.
- Follow PEP 8 conventions.
- Use meaningful names for variables, functions, and modules.
- Document important functions and decisions.
- Avoid data leakage.
- Use reproducible experiments and fixed random seeds where appropriate.
- Validate inputs and outputs.
- Write tests for critical components.
- Keep dependencies and configuration explicit.

## 5. Project Architecture

- `configs/`: Project configuration.
- `data/raw/`: Original datasets.
- `data/processed/`: Processed datasets.
- `docs/`: Project documentation and technical decisions.
- `notebooks/`: Exploratory analysis and experiments.
- `reports/`: Figures and evaluation reports.
- `src/`: Reusable application and ML code.
- `tests/`: Automated tests.
- `models/`: Saved model artifacts.
- `.github/workflows/`: CI workflows.

## 6. Machine Learning Workflow

1. Problem Definition
2. Dataset Collection
3. Exploratory Data Analysis
4. Data Quality Assessment
5. Data Preprocessing
6. Baseline Model
7. Model Development
8. Model Evaluation
9. Error Analysis
10. Model Improvement
11. Model Packaging
12. Testing
13. Documentation and GitHub Release
14. Deployment (planned extension)

## 7. Git Workflow

- `main` must contain stable project versions.
- Use feature branches for substantial development tasks.
- Write small, meaningful commits.
- Follow Conventional Commits.

Examples:
- `docs: define project objectives`
- `feat: implement preprocessing pipeline`
- `test: add model validation tests`
- `fix: handle missing values correctly`

## 8. GitHub Repository Rules

- Never commit secrets or credentials.
- Do not commit virtual environments.
- Do not commit unnecessary generated files.
- Do not publish datasets without checking their licenses.
- Document dataset sources and download instructions.
- Keep README accurate and reproducible.

## 9. Collaboration and Learning Workflow

For each development phase:

1. Understand the objective and required concepts.
2. Review relevant theory.
3. Design the implementation.
4. Implement the solution.
5. Review code quality and correctness.
6. Run tests and verify results.
7. Document decisions and findings.
8. Commit completed work.

The learner should understand and be able to explain all important implementation decisions.

## 10. Definition of Done

A development task is complete when:

- The implementation works as expected.
- Relevant tests pass.
- The code follows project conventions.
- Results are reproducible where applicable.
- Documentation is updated.
- Changes are committed to Git.

## 11. Scope Control

Advanced deployment and infrastructure features will be introduced only after the core machine learning workflow is understood and validated.

The project should remain educational, practical, maintainable, and appropriate for a junior Machine Learning Engineer portfolio.
