# Automated Medical Diagnosis and Personalized Treatment Optimization

A layered clinical decision-support framework for **automated medical diagnosis** and **personalized treatment optimization** using:

- Classical Reinforcement Learning: **SARSA, Q-Learning, Monte Carlo**
- Deep Reinforcement Learning: **DQN / Double DQN**
- Inverse Reinforcement Learning: **reward inference from expert trajectories**
- Multi-Agent Reinforcement Learning: **coordinated care agents**
- Safety and explainability layers for healthcare decision support

> This project is a **clinical decision-support system**, not an autonomous replacement for doctors.

---

## Project Goal

The system is designed to:

- analyze patient data for diagnosis support
- recommend personalized treatment actions over time
- learn from sequential clinical outcomes
- align decisions with clinician behavior
- scale from simple tabular learning to deep, multi-agent healthcare environments

---

## Full System Architecture

```text
              +----------------------------------+
              |      Data Acquisition Layer      |
              |----------------------------------|
              | EHR | Labs | Vitals | Imaging    |
              | Notes | Medication | History     |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              |     Data Processing Layer        |
              |----------------------------------|
              | Cleaning | Normalization         |
              | Missing Value Handling           |
              | Temporal Sequencing              |
              | Feature Encoding                 |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              |   Patient State Representation   |
              |----------------------------------|
              | Structured Feature Encoder       |
              | Image Encoder                    |
              | Clinical Text Encoder            |
              | Temporal State Builder           |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              |   MDP Environment Formulation    |
              |----------------------------------|
              | State Space S                    |
              | Action Space A                   |
              | Transition Model P               |
              | Reward Function R                |
              | Policy Objective pi(a|s)         |
              +----------------+-----------------+
                               |
         +---------------------+----------------------+
         |                     |                      |
         v                     v                      v
+----------------------+ +----------------------+ +----------------------+
| Classical RL Layer   | | Deep RL Layer        | | Expert Learning      |
|----------------------| |----------------------| |----------------------|
| SARSA                | | DQN / DDQN / PPO     | | Inverse RL           |
| Q-Learning           | | High-dim state input | | Clinician behavior   |
| Monte Carlo Eval     | | Diagnosis+treatment  | | Reward inference     |
+----------+-----------+ +----------+-----------+ +----------+-----------+
           |                        |                         |
           +------------------------+-------------------------+
                                    |
                                    v
                  +----------------------------------+
                  | Clinical Decision Engine         |
                  |----------------------------------|
                  | Diagnosis Support                |
                  | Treatment Recommendation         |
                  | Dose/Timing Adaptation           |
                  | Risk-Aware Action Ranking        |
                  +----------------+-----------------+
                                   |
                                   v
                  +----------------------------------+
                  | Multi-Agent Coordination Layer   |
                  |----------------------------------|
                  | Hospital Resource Agent          |
                  | ICU Monitoring Agent             |
                  | Robotic Assistance Agent         |
                  | Collaborative Decision Fusion    |
                  +----------------+-----------------+
                                   |
                                   v
                  +----------------------------------+
                  | Evaluation, Safety, Explainability|
                  |----------------------------------|
                  | Reward Analysis                  |
                  | Accuracy / Outcome Metrics       |
                  | Safety Constraints               |
                  | Clinician Agreement              |
                  | Policy Interpretability          |
                  +----------------------------------+
```

---

## Two-Stage Pipeline

### Stage 1: Diagnosis Training from CSV
The first stage reads `patient_dataset.csv` and trains diagnosis models to predict:

- `disease_name`
- `disease_severity`

This stage compares multiple diagnosis models such as:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVC
- KNN

The best diagnosis model is selected based on classification accuracy.

### Stage 2: Reinforcement Learning Treatment Module
The second stage uses the diagnosed patient state and severity estimate as the starting point for treatment optimization.

This stage includes:

- **SARSA** as an on-policy baseline
- **Q-Learning** as an off-policy baseline
- **Monte Carlo** for return estimation and comparison
- **Double DQN** for higher-dimensional policy learning
- **Inverse RL** for clinician-aligned reward shaping
- **Safety Layer** for contraindication filtering
- **Clinical Decision Engine** for final recommendation generation

---

## Data Flow

1. Patient records are loaded from CSV
2. Features are cleaned, normalized, and encoded
3. Diagnosis models are trained and compared
4. The best diagnosis model predicts disease and severity
5. RL agents interact with a patient-state environment
6. Safety rules filter unsafe actions
7. The decision engine ranks and explains treatment actions

---

## Core Modules

### 1. Data Acquisition Layer
Collects:

- Electronic Health Records
- Laboratory reports
- Vital signs
- Imaging
- Medication history
- Disease history
- Clinician notes
- Treatment timelines
- ICU monitoring streams

### 2. Data Processing Layer
Handles:

- duplicate removal
- missing value handling
- normalization
- categorical encoding
- temporal sequencing
- outcome extraction
- patient episode construction

Example patient episode:

```text
State_1 -> Action_1 -> Reward_1 -> State_2 -> Action_2 -> Reward_2 -> State_3
```

### 3. Patient State Representation
Transforms multimodal patient information into a compact state representation using:

- structured feature encoder
- image encoder
- text encoder
- temporal state builder

### 4. MDP Environment
Defines:

- **State space (S)**: patient condition at time `t`
- **Action space (A)**: diagnosis or treatment decisions
- **Transition function (P)**: patient state evolution
- **Reward function (R)**: health improvement, safety, cost, and efficiency
- **Policy**: action selection for the current patient state

### 5. Classical RL Layer
Includes:

- **SARSA**
- **Q-Learning**
- **Monte Carlo Evaluation**

These provide interpretable baselines before moving to deep RL.

### 6. Deep RL Layer
Used for high-dimensional patient states where tabular methods are insufficient.

Possible methods:

- DQN
- Double DQN
- Dueling DQN
- PPO
- Actor-Critic

### 7. Inverse RL Layer
Learns hidden reward structure from clinician trajectories to improve policy alignment.

### 8. Multi-Agent RL Layer
Supports collaborative clinical settings such as:

- diagnosis agent
- treatment agent
- resource allocation agent
- ICU monitoring agent
- robotic assistance agent

### 9. Safety Layer
Prevents unsafe or contraindicated recommendations.

Examples:

- block contraindicated treatments
- block unsafe discharge
- add clinician override
- enforce conservative rules for high-risk cases

### 10. Explainability Layer
Provides:

- top contributing features
- action-value scores
- clinician alignment signals
- recommendation reasoning

### 11. Evaluation Layer
Measures:

#### Diagnosis Metrics
- accuracy
- precision
- recall
- F1-score
- AUROC
- calibration

#### Treatment Metrics
- cumulative reward
- success rate
- adverse event rate
- readmission reduction
- policy consistency

#### RL Metrics
- episode return
- convergence rate
- Q-value stability
- reward sensitivity
- exploration vs exploitation balance

#### Safety Metrics
- unsafe action rate
- constraint violation rate
- high-risk patient stability

---

## Deployment View

```text
Hospital Data Systems
        |
        v
Data Integration API
        |
        v
Preprocessing + Feature Service
        |
        +----------------------------+
        |                            |
        v                            v
Diagnosis Inference Service     RL Treatment Policy Service
        |                            |
        +-------------+--------------+
                      |
                      v
            Safety + Rule Validation
                      |
                      v
            Clinical Decision Dashboard
                      |
                      v
              Doctor Review / Override
```

---

## Output Files

The pipeline can generate outputs such as:

- `diagnosis_model_comparison.csv`
- `diagnosis_predictions.csv`
- `diagnosis_classification_report.json`
- `rl_agent_comparison.csv`
- `pipeline_summary.json`
- `sample_recommendation.txt`

---

## Recommended Interpretation of Final Score

This project should not report a single plain “accuracy” for the full pipeline unless the meaning is defined clearly.

A better summary metric is a **composite system score**, for example:

```text
Total_Model_Score =
0.50 * Diagnosis_Accuracy
+ 0.20 * (1 - Normalized_Severity_Error)
+ 0.20 * Safe_Action_Rate
+ 0.10 * Treatment_Success_Rate
```

This is more honest than calling the entire end-to-end system “accuracy”.

---

## Limitations

- The dataset may still be synthetic or semi-synthetic
- Diagnosis labels may overlap and reduce class separability
- RL treatment policies depend heavily on reward design
- The framework is suited for research and experimentation, not direct clinical deployment
- Diagnosis should not be treated as purely an RL problem

---

## Future Improvements

- Replace baseline diagnosis models with **XGBoost / CatBoost**
- Add image and text encoders for real multimodal diagnosis
- Improve reward shaping using more realistic clinical constraints
- Add stronger clinician-trajectory data for IRL
- Expand MARL coordination for ICU and surgical robotics
- Validate on real-world longitudinal healthcare datasets

---

## Final Architecture Statement

The proposed system follows a layered healthcare AI architecture in which multimodal patient data are transformed into sequential patient state representations and modeled through a Markov Decision Process. Classical reinforcement learning methods such as SARSA and Q-learning serve as interpretable baselines, Monte Carlo methods support return-based policy evaluation, and Deep Reinforcement Learning handles high-dimensional diagnosis and treatment state spaces. Inverse Reinforcement Learning is integrated to infer clinician-aligned reward structures from expert behavior, while Multi-Agent Reinforcement Learning extends the framework to collaborative care settings such as ICU coordination, hospital resource allocation, and robotic assistance. A safety-aware clinical decision engine combines these components to produce explainable and risk-constrained recommendations for automated medical diagnosis and personalized treatment optimization.
