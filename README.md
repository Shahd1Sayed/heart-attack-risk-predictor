<h1 align="center">🫀 Heart Attack Risk Predictor</h1>

<p align="center">
  <strong>An AI-powered clinical decision-support tool that predicts a patient's heart attack risk level (High, Moderate, or Low) based on 8 vital signs and cardiac biomarkers, served as a real-time web application powered by a Random Forest classifier and FastAPI.</strong>
</p>

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Medical Background](#-medical-background)
- [Tech Stack](#-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset Description](#-dataset-description)
- [Feature Importance](#-feature-importance)
- [Team Members](#-team-members)

---

## ✨ Key Features

- **Real-Time Risk Prediction** — Enter 8 patient vitals and receive an instant heart attack risk classification (High / Moderate / Low) with confidence probabilities.
- **Production-Grade REST API** — Built with FastAPI, featuring automatic request validation via Pydantic schemas, interactive Swagger docs, and ASGI-based async performance.
- **Modern UI** — A dark-themed, responsive frontend with animated gradient blobs, color-coded risk badges (🔴 High / 🟡 Moderate / 🟢 Low), and animated probability bars.
- **Bundled Model Artifact** — The trained model, scaler, and label encoders are serialized into a single `model.pkl` for zero-configuration deployment.
- **Clinically Interpretable** — Feature importance rankings show exactly which vitals drive predictions, aligning with established cardiology literature.

---

## 🏥 Medical Background

### What Is a Heart Attack?

A **heart attack** (myocardial infarction, MI) occurs when blood flow to a section of the heart muscle is blocked for an extended period, causing the heart tissue to die. It is the leading cause of death globally, accounting for approximately **17.9 million deaths per year** according to the World Health Organization (WHO).

### Clinical Biomarkers Used in This Model

This application leverages the same vital signs and biomarkers that emergency physicians use to triage cardiac patients:

| Biomarker | Clinical Significance | Normal Range |
|---|---|---|
| **Age** | Risk increases significantly after age 45 (men) and 55 (women). Aging contributes to arterial stiffness, plaque buildup, and decreased cardiac output. | N/A |
| **Gender** | Men have a statistically higher risk of MI at younger ages. Post-menopausal women see a sharp increase in risk due to declining estrogen levels. | Male / Female |
| **Heart Rate** | Resting tachycardia (>100 bpm) is associated with increased cardiovascular mortality. Bradycardia (<60 bpm) may indicate conduction abnormalities. | 60–100 bpm |
| **Systolic Blood Pressure** | Elevated systolic BP (hypertension) is a primary modifiable risk factor. It places chronic stress on arterial walls, accelerating atherosclerosis. | 90–120 mmHg |
| **Diastolic Blood Pressure** | Elevated diastolic BP reduces coronary perfusion pressure and is independently associated with increased MI risk, especially in younger adults. | 60–80 mmHg |
| **Blood Sugar (Glucose)** | Hyperglycemia (diabetes) accelerates endothelial dysfunction and atherosclerotic plaque formation. Diabetic patients have a 2–4× higher risk of cardiovascular events. | 70–140 mg/dL |
| **CK-MB (Creatine Kinase-MB)** | A cardiac enzyme released into the bloodstream when the heart muscle is damaged. Elevated CK-MB (>5 ng/mL) within 4–8 hours of chest pain is a strong indicator of myocardial injury. | 0–5 ng/mL |
| **Troponin** | The **gold standard** biomarker for diagnosing acute MI. Cardiac troponins (cTnI / cTnT) are structural proteins released during myocardial necrosis. Even slight elevations (>0.04 ng/mL) are diagnostic. Troponin levels peak at 12–24 hours post-MI. | <0.04 ng/mL |

### Why Troponin and CK-MB Dominate

In our trained model, **Troponin** (43.3% importance) and **CK-MB** (20.7% importance) together account for **64%** of the model's decision-making. This is medically consistent — cardiologists rely heavily on these two biomarkers to diagnose and classify MI severity. The `Risk_Level` labels in the dataset are likely derived from clinical threshold rules applied to these biomarkers, which explains the model's exceptionally high accuracy (~98.6%).

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance ASGI web framework with automatic OpenAPI docs |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | Lightning-fast ASGI server for production HTTP serving |
| **ML Model** | [scikit-learn](https://scikit-learn.org/) `RandomForestClassifier` | Ensemble learning method for multi-class classification |
| **Preprocessing** | [scikit-learn](https://scikit-learn.org/) `StandardScaler`, `LabelEncoder` | Feature normalization and categorical encoding |
| **Model Comparison** | [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting framework used for benchmarking |
| **Data Handling** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | DataFrame manipulation and numerical computation |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) | Data validation and serialization for API request/response |
| **Serialization** | Python `pickle` | Model persistence and bundling |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Responsive single-page application with glassmorphism design |

---

## 🤖 Machine Learning Pipeline

The training pipeline (`research_and_experiments/train_model.py`) follows a rigorous, multi-stage workflow:

### Stage 1 — Data Loading & Inspection
```
Raw dataset: 1,319 rows × 11 columns
Target variable: Risk_Level (High / Moderate / Low)
```

### Stage 2 — Outlier Cleaning
Rows with clinically implausible values are removed using evidence-based physiological bounds:

| Feature | Valid Range | Outliers Removed |
|---|---|---|
| Heart rate | 20–300 bpm | 3 (e.g., 1 bpm) |
| Systolic BP | 50–300 mmHg | 1 |
| Age | 1–120 years | 0 |
| CK-MB | 0–500 ng/mL | 0 |
| Troponin | 0–50 ng/mL | 0 |

**Result:** 1,319 → 1,315 rows (4 outliers removed)

### Stage 3 — Encoding & Scaling
- **Gender**: `LabelEncoder` (Female → 0, Male → 1)
- **Risk_Level**: `LabelEncoder` (High → 0, Low → 1, Moderate → 2)
- **All features**: `StandardScaler` (zero mean, unit variance)

### Stage 4 — Model Comparison (5-Fold Stratified CV)
Four models were rigorously evaluated to ensure the best performer was selected:

| Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| **Random Forest (default)** | **98.63%** | ±0.52% |
| Random Forest (balanced) | 98.48% | ±0.59% |
| XGBoost (default) | 98.48% | ±0.48% |
| XGBoost (balanced) | 98.63% | ±0.52% |

### Stage 5 — Final Training & Export
The winning model (**Random Forest, 200 estimators**) is retrained on the full cleaned dataset and bundled into `model.pkl` alongside the scaler and encoders.

---

## 📁 Project Structure

```
heart-attack-risk-predictor/
│
├── app.py                          # FastAPI backend — loads model, serves predictions & frontend
├── model.pkl                       # Serialized bundle: RF model + StandardScaler + LabelEncoders
├── requirements.txt                # Production Python dependencies
│
├── static/                         # Frontend assets
│   └── index.html                  # Single-page app with glassmorphism UI
│
└── research_and_experiments/       # Exploratory analysis & training (not needed for production)
    ├── heart_attack.ipynb          # Jupyter notebook with EDA and visualizations
    ├── Heart_Attack_Risk_Levels_Dataset.csv   # Raw dataset (1,319 patient records)
    ├── train_model.py              # Full training pipeline with diagnostics
    ├── save_as_pkl.py              # Script to bundle model artifacts into model.pkl
    ├── output.txt                  # Training output logs
    ├── random_forest_model.joblib  # Individual model artifact (legacy)
    ├── label_encoder_gender.joblib # Individual encoder artifact (legacy)
    └── label_encoder_risk_level.joblib  # Individual encoder artifact (legacy)
```

### File Descriptions

| File | Size | Description |
|---|---|---|
| `app.py` | 3 KB | The core application server. Loads the model bundle at startup, exposes a `POST /predict` endpoint, and serves the static frontend at `/`. |
| `model.pkl` | 2.6 MB | A Python pickle bundle containing four objects: the trained `RandomForestClassifier`, the fitted `StandardScaler`, the `LabelEncoder` for gender, and the `LabelEncoder` for risk level. |
| `requirements.txt` | <1 KB | Pinned production dependencies: `fastapi`, `uvicorn`, `scikit-learn`, `numpy`, `pydantic`. |
| `static/index.html` | ~9 KB | A self-contained HTML/CSS/JS frontend with dark theme, animated gradient background, input validation, and dynamic result rendering. |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.9+ | `python --version` |
| pip | 21.0+ | `pip --version` |

### Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/Shahd1Sayed/heart-attack-risk-predictor.git
```

**2. Navigate to the project directory**
```bash
cd heart-attack-risk-predictor
```

**3. Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Verify the model file exists**
```bash
# Ensure model.pkl is present in the root directory
ls model.pkl        # macOS/Linux
dir model.pkl       # Windows
```

> **Note:** If `model.pkl` is missing (e.g., excluded by `.gitignore` due to its size), you can regenerate it by running the training pipeline:
> ```bash
> cd research_and_experiments
> pip install pandas xgboost   # additional training dependencies
> python train_model.py
> mv model.pkl ../             # move the generated model to root
> cd ..
> ```

---

## ▶️ Usage

### Start the server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Access the application
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Using the Web Interface
1. Fill in the 8 patient vitals in the form fields.
2. Click **"Predict Risk Level"**.
3. View the color-coded result badge and probability distribution bars.

### Using the API directly (cURL)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "gender": "Male",
    "heart_rate": 66,
    "systolic_bp": 160,
    "diastolic_bp": 83,
    "blood_sugar": 160,
    "ck_mb": 1.8,
    "troponin": 0.012
  }'
```

### Example Response
```json
{
  "risk_level": "Moderate",
  "probabilities": {
    "High": 0.045,
    "Moderate": 0.925,
    "Low": 0.03
  }
}
```

---

### Endpoints

#### `GET /`
Serves the frontend HTML application.

| Parameter | Value |
|---|---|
| **Method** | `GET` |
| **Response** | `text/html` — The single-page application |
| **Status** | `200 OK` |

---

#### `POST /predict`
Accepts patient vitals and returns the predicted heart attack risk level with class probabilities.

**Request Body** (`application/json`):

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| `age` | `integer` | ✅ | Patient's age in years | `63` |
| `gender` | `string` | ✅ | `"Male"` or `"Female"` | `"Male"` |
| `heart_rate` | `integer` | ✅ | Resting heart rate in bpm | `66` |
| `systolic_bp` | `integer` | ✅ | Systolic blood pressure in mmHg | `160` |
| `diastolic_bp` | `integer` | ✅ | Diastolic blood pressure in mmHg | `83` |
| `blood_sugar` | `integer` | ✅ | Fasting blood sugar in mg/dL | `160` |
| `ck_mb` | `float` | ✅ | CK-MB enzyme level in ng/mL | `1.8` |
| `troponin` | `float` | ✅ | Cardiac troponin level in ng/mL | `0.012` |

**Response Body** (`application/json`):

| Field | Type | Description |
|---|---|---|
| `risk_level` | `string` | Predicted class: `"High"`, `"Moderate"`, or `"Low"` |
| `probabilities` | `object` | Confidence scores for each class (sum to 1.0) |

**Response Status Codes:**

| Code | Meaning |
|---|---|
| `200 OK` | Prediction returned successfully |
| `422 Unprocessable Entity` | Invalid or missing request fields (Pydantic validation error) |

---

## 📊 Model Performance

### Cross-Validation Results (5-Fold Stratified)

| Metric | Value |
|---|---|
| **Best Model** | Random Forest (200 estimators) |
| **Mean CV Accuracy** | 98.63% |
| **Standard Deviation** | ±0.52% |
| **Train Accuracy** | 100.00% |
| **Train-Test Gap** | ~1.4% (minor overfitting) |

### Class Distribution in Dataset

| Risk Level | Count | Percentage |
|---|---|---|
| High | 809 | 61.5% |
| Low | 275 | 20.9% |
| Moderate | 231 | 17.6% |

### Accuracy Transparency Note

> ⚠️ **Important:** The high accuracy (~98.6%) is attributable to the dataset's construction. Analysis revealed that the `Risk_Level` labels are strongly correlated with — and likely derived from — clinical threshold rules applied to **Troponin** and **CK-MB** values. Removing these two features drops accuracy to **62.17%**. This means the model is effectively learning decision boundaries that mirror clinical diagnostic criteria, which is valid for a decision-support tool but should not be interpreted as the model discovering novel medical insights.

---

## 📋 Dataset Description

- **Source:** Heart Attack Risk Levels Dataset
- **Records:** 1,319 patient observations (1,315 after outlier removal)
- **Features:** 8 clinical input variables + 1 target variable
- **Target Variable:** `Risk_Level` — a 3-class categorical label

### Full Schema

| Column | Data Type | Range / Values | Description |
|---|---|---|---|
| `Age` | Integer | 14–98 | Patient age in years |
| `Gender` | Binary | 0 (Female), 1 (Male) | Biological sex |
| `Heart rate` | Integer | 20–300 | Resting heart rate (bpm) |
| `Systolic blood pressure` | Integer | 50–300 | Systolic BP (mmHg) |
| `Diastolic blood pressure` | Integer | 20–200 | Diastolic BP (mmHg) |
| `Blood sugar` | Integer | 20–700 | Fasting blood glucose (mg/dL) |
| `CK-MB` | Float | 0–500 | Creatine Kinase-MB isoenzyme (ng/mL) |
| `Troponin` | Float | 0–50 | Cardiac troponin concentration (ng/mL) |
| `Risk_Level` | Categorical | High / Moderate / Low | **Target** — Heart attack risk classification |

---

## 🔍 Feature Importance

The following chart shows how much each feature contributes to the model's predictions (measured by Gini impurity decrease in the Random Forest):

```
Troponin                       ██████████████████████  43.3%
CK-MB                          ██████████             20.7%
Systolic blood pressure        ███████                13.4%
Blood sugar                    ██████                 11.1%
Age                            ██                      4.3%
Diastolic blood pressure       ██                      4.0%
Heart rate                     █                       2.5%
Gender                         ▏                       0.7%
```

### Clinical Interpretation

1. **Troponin (43.3%)** — Aligns with its role as the gold-standard biomarker for MI diagnosis. The European Society of Cardiology (ESC) guidelines classify MI severity primarily based on troponin thresholds.
2. **CK-MB (20.7%)** — Historically the primary cardiac biomarker before high-sensitivity troponin assays became standard. Still used as a complementary marker.
3. **Blood Pressure (17.4% combined)** — Hypertension is the single most prevalent modifiable risk factor for cardiovascular disease.
4. **Blood Sugar (11.1%)** — Reflects the well-established link between diabetes/metabolic syndrome and cardiovascular risk.
5. **Age (4.3%)** — A non-modifiable risk factor captured by the Framingham Risk Score.
6. **Gender (0.7%)** — Low importance suggests the dataset may not fully capture sex-based cardiovascular differences.

---


### Development Setup

To work on the training pipeline, install the additional research dependencies:
```bash
pip install pandas xgboost jupyter matplotlib seaborn
```

---

## 👥 Team Members

| Name |
|---|
| **Shahd Sayed** |
| **Shahd Mohammed** |
---
