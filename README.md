# AI-Driven Cloud Infrastructure Optimization & Predictive Maintenance Engine
### *Interactive Multi-Algorithm Benchmarking Pipeline & Automation Framework*

Welcome to the **Next-Gen Cloud-Ops AI Engine**. This repository hosts a production-grade 4-Tier Data Pipeline and web-accessible dashboard that addresses corporate cloud resource misallocations. The system leverages descriptive data analytics to isolate financial budget waste ("Zombie Nodes") and deploys parallel supervised machine learning algorithms to predict system crash events before they compromise active client SLAs.

---

## 🏗️ System Core Architecture
The software framework is decoupled into four modular, production-ready tiers:
1. **Data Ingestion Pipeline:** Uses vectorized random engines to ingest and validate continuous time-series telemetry log matrices.
2. **Feature Engineering Layer:** Implements logical data constraints (`np.where`) to programmatically extract cluster performance profiles and calculate operational financial leakage.
3. **Multi-Model ML Benchmarking Engine:** Runs training runs in parallel to evaluate Linear models, Ensemble architectures, and Hyperplane boundaries (`Random Forest` vs. `SVM` vs. `Logistic Regression`).
4. **Graphical User Dashboard:** A web-accessible micro-frontend interface built via Streamlit to handle real-time simulation controls and risk inference mapping.

---

## 📁 Repository Directory Structure
```text
AI_Data_Analyst_Project/
├── app.py                      # Core Full-Stack Web Application (Frontend + AI Engine)
├── ai_analytics_pipeline.py     # Background Data Processing Script
├── cloud_telemetry_final.csv    # Cleaned Tabular Master Telemetry Dataset (3,000 Rows)
├── AI_Project_Final_Report.md   # Formal Academic Capstone Thesis Documentation
└── README.md                    # System Deployment Manual (This File)
```

---

## 🛠️ Developer Setup & Deployment Guide
Follow these precise terminal instructions to install dependencies and deploy the interactive host server locally on your workstation.

### 1. Environment Configurations
Ensure your local machine has Python 3.10+ configured. Open the system command line interface terminal window and install the standard enterprise data science libraries:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

### 2. Launch the Interactive Dashboard Web App
Navigate to the root directory folder and execute the Streamlit host engine long-form command:
```bash
python -m streamlit run app.py
```
*The local host web server port will boot automatically within 3 seconds, launching the live interface inside your active web browser tab (`http://localhost:8501`).*

---

## 🔮 Active Live Demonstration Instructions
When showcasing this platform to project evaluators, use the following interactive validation steps:
* **Real-Time Stress Testing:** Click and slide the **CPU** and **Memory** sliders in the left panel. Watch the yellow indicator marker shift dynamically across the live scatter plot boundaries.
* **Triggering the AI Race:** Scroll down and click the **"Execute Algorithmic Race"** button. The application will immediately train three distinct machine learning models and render side-by-side performance comparison charts.
* **Self-Healing Simulation:** Push the simulation controls past safe operational thresholds (>85% CPU and >50GB RAM). The green safety check block will instantly change to a deep crimson red alert, triggering an automated system stabilization container log routine.

---
*Developed inside the IBM Bob Agentic AI Environment as a CSE Final Year Major Project Capstone.*