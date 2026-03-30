import os
import ast
import json
import random
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\Users\saivi\healthcare rl"
DATASET_PATH = os.path.join(BASE_DIR, "patient_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
N_RL_EPISODES = 300
N_EVAL_EPISODES = 100

# ============================================================
# DATASET-ALIGNED ACTIONS
# ============================================================
ACTIONS = [
    "monitor",
    "lifestyle_modification",
    "start_medication",
    "adjust_medication",
    "order_followup_tests",
    "refer_specialist",
    "admit_hospital",
    "discharge",
]

DISEASE_ACTION_MAP = {
    "Type 2 Diabetes": ["lifestyle_modification", "start_medication", "order_followup_tests"],
    "Hypertension": ["lifestyle_modification", "start_medication", "order_followup_tests"],
    "Coronary Artery Disease": ["start_medication", "refer_specialist", "admit_hospital"],
    "COPD": ["start_medication", "refer_specialist", "admit_hospital"],
    "Chronic Kidney Disease": ["start_medication", "order_followup_tests", "refer_specialist"],
    "Obesity": ["lifestyle_modification", "order_followup_tests", "monitor"],
    "Depression": ["start_medication", "refer_specialist", "monitor"],
    "Asthma": ["start_medication", "monitor", "refer_specialist"],
    "Liver Cirrhosis": ["start_medication", "refer_specialist", "admit_hospital"],
    "Anemia": ["start_medication", "order_followup_tests", "monitor"],
}

CONTRAINDICATIONS = {
    "Chronic Kidney Disease": ["discharge"],
    "Coronary Artery Disease": ["discharge"],
    "COPD": ["discharge"],
    "Liver Cirrhosis": ["discharge"],
}

# ============================================================
# HELPERS
# ============================================================
def parse_feature_vector(val) -> List[float]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return []


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def build_total_model_accuracy(diagnosis_accuracy: float,
                               severity_mae: float,
                               safe_action_rate: float,
                               treatment_success_rate: float) -> float:
    severity_score = max(0.0, 1.0 - severity_mae)
    total = (
        0.50 * diagnosis_accuracy
        + 0.20 * severity_score
        + 0.20 * safe_action_rate
        + 0.10 * treatment_success_rate
    )
    return float(total)


# ============================================================
# STAGE 1: DIAGNOSIS TRAINING + MODEL COMPARISON
# ============================================================
class DiagnosisStage:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.label_encoder = LabelEncoder()
        self.best_model_name: Optional[str] = None
        self.best_pipeline = None
        self.best_accuracy = -1.0
        self.best_metrics: Dict[str, float] = {}
        self.feature_columns: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.regressor = None
        self.regression_pipeline = None

    def load(self):
        df = pd.read_csv(self.csv_path)
        df["feature_vector"] = df["feature_vector"].apply(parse_feature_vector)
        self.df = df
        return df

    def prepare_features(self):
        assert self.df is not None
        df = self.df.copy()

        drop_cols = [
            "patient_id",
            "disease_name",
            "icd10_code",
            "body_system",
            "disease_severity",
            "primary_causes",
            "symptoms_present",
            "prevention_plan",
            "recommended_medications",
        ]

        if df["feature_vector"].map(len).max() > 0:
            max_len = int(df["feature_vector"].map(len).max())
            fv_df = pd.DataFrame(
                df["feature_vector"].tolist(),
                columns=[f"fv_{i}" for i in range(max_len)]
            )
            df = pd.concat([df.drop(columns=["feature_vector"]), fv_df], axis=1)
        else:
            df = df.drop(columns=["feature_vector"])

        X = df.drop(columns=drop_cols, errors="ignore")
        y_cls = df["disease_name"].copy()
        y_reg = df["disease_severity"].astype(float).copy()

        self.feature_columns = list(X.columns)
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        return X, y_cls, y_reg

    def build_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )
        return preprocessor

    def compare_models(self):
        X, y_cls, y_reg = self.prepare_features()
        y_encoded = self.label_encoder.fit_transform(y_cls)

        X_train, X_test, y_train, y_test, sev_train, sev_test = train_test_split(
            X, y_encoded, y_reg,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y_encoded,
        )

        preprocessor = self.build_preprocessor()

        models = {
            "LogisticRegression": LogisticRegression(max_iter=3000, random_state=SEED),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
            "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
            "SVC": SVC(kernel="rbf", probability=True, random_state=SEED),
            "KNN": KNeighborsClassifier(n_neighbors=7),
        }

        results = []
        for name, model in models.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results.append({"model": name, "accuracy": float(acc)})
            if acc > self.best_accuracy:
                self.best_accuracy = float(acc)
                self.best_model_name = name
                self.best_pipeline = pipe
                self.best_metrics = {
                    "diagnosis_accuracy": float(acc)
                }

        # severity regressor
        regressor = RandomForestRegressor(n_estimators=300, random_state=SEED)
        reg_pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ])
        reg_pipe.fit(X_train, sev_train)
        sev_preds = reg_pipe.predict(X_test)
        sev_mae = mean_absolute_error(sev_test, sev_preds)

        self.regressor = regressor
        self.regression_pipeline = reg_pipe
        self.best_metrics["severity_mae"] = float(sev_mae)

        result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
        result_df.to_csv(os.path.join(OUTPUT_DIR, "diagnosis_model_comparison.csv"), index=False)

        test_out = X_test.copy()
        test_out["true_disease"] = self.label_encoder.inverse_transform(y_test)
        test_out["pred_disease"] = self.label_encoder.inverse_transform(self.best_pipeline.predict(X_test))
        test_out["true_severity"] = sev_test.values
        test_out["pred_severity"] = self.regression_pipeline.predict(X_test)
        test_out.to_csv(os.path.join(OUTPUT_DIR, "diagnosis_predictions.csv"), index=False)

        report = classification_report(
            y_test,
            self.best_pipeline.predict(X_test),
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        with open(os.path.join(OUTPUT_DIR, "diagnosis_classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "sev_train": sev_train,
            "sev_test": sev_test,
            "comparison": result_df,
            "best_model_name": self.best_model_name,
            "best_accuracy": self.best_accuracy,
            "severity_mae": float(sev_mae),
        }

    def predict_patient(self, patient_row: pd.Series) -> Dict:
        assert self.best_pipeline is not None
        assert self.regression_pipeline is not None

        row = pd.DataFrame([patient_row]).copy()
        row["feature_vector"] = row["feature_vector"].apply(parse_feature_vector)
        if row["feature_vector"].map(len).max() > 0:
            max_len = int(row["feature_vector"].map(len).max())
            fv_df = pd.DataFrame(
                row["feature_vector"].tolist(),
                columns=[f"fv_{i}" for i in range(max_len)]
            )
            row = pd.concat([row.drop(columns=["feature_vector"]), fv_df], axis=1)
        else:
            row = row.drop(columns=["feature_vector"], errors="ignore")

        X = row[self.feature_columns].copy()
        pred_idx = self.best_pipeline.predict(X)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        pred_severity = float(self.regression_pipeline.predict(X)[0])
        probs = self.best_pipeline.predict_proba(X)[0] if hasattr(self.best_pipeline, "predict_proba") else None

        return {
            "predicted_disease": pred_label,
            "predicted_severity": clip01(pred_severity),
            "probabilities": probs,
        }


# ============================================================
# RL STATE + DATASET-ALIGNED ENVIRONMENT
# ============================================================
@dataclass
class PatientState:
    features: np.ndarray
    diagnosis: str
    severity: float
    patient_row: pd.Series
    step: int = 0
    episode_done: bool = False

    def to_vector(self) -> np.ndarray:
        return np.append(self.features, [self.severity])

    def discretize(self, bins: int = 5) -> tuple:
        discrete = tuple(int(np.clip(v * bins, 0, bins - 1)) for v in self.features)
        sev_bin = int(np.clip(self.severity * bins, 0, bins - 1))
        return discrete + (sev_bin,)


class DatasetHealthcareEnv:
    MAX_STEPS = 10

    def __init__(self, df: pd.DataFrame, diagnosis_stage: DiagnosisStage, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.diagnosis_stage = diagnosis_stage
        self.rng = np.random.default_rng(seed)
        self.state: Optional[PatientState] = None
        self.sample_idx: Optional[int] = None
        self.feature_cols_for_env = [
            "age", "bmi", "fruit_veg_servings_per_day", "sugar_intake_g_per_day",
            "sodium_intake_mg_per_day", "screen_time_hours", "outdoor_time_hours",
            "heart_rate", "blood_pressure_systolic", "oxygen_saturation",
            "temperature_c", "glucose_mg_dl", "wbc", "creatinine",
            "hemoglobin", "sodium_meq", "potassium"
        ]
        self.feature_norms = {
            "age": 100.0,
            "bmi": 50.0,
            "fruit_veg_servings_per_day": 10.0,
            "sugar_intake_g_per_day": 200.0,
            "sodium_intake_mg_per_day": 5000.0,
            "screen_time_hours": 16.0,
            "outdoor_time_hours": 8.0,
            "heart_rate": 180.0,
            "blood_pressure_systolic": 220.0,
            "oxygen_saturation": 100.0,
            "temperature_c": 45.0,
            "glucose_mg_dl": 300.0,
            "wbc": 25.0,
            "creatinine": 6.0,
            "hemoglobin": 20.0,
            "sodium_meq": 170.0,
            "potassium": 8.0,
        }

    def _row_to_features(self, row: pd.Series) -> np.ndarray:
        vec = []
        for col in self.feature_cols_for_env:
            val = float(row[col]) if col in row and not pd.isna(row[col]) else 0.0
            denom = self.feature_norms[col]
            vec.append(np.clip(val / denom, 0.0, 1.0))
        return np.array(vec, dtype=np.float32)

    def reset(self) -> PatientState:
        self.sample_idx = int(self.rng.integers(0, len(self.df)))
        row = self.df.iloc[self.sample_idx].copy()
        pred = self.diagnosis_stage.predict_patient(row)
        self.state = PatientState(
            features=self._row_to_features(row),
            diagnosis=pred["predicted_disease"],
            severity=pred["predicted_severity"],
            patient_row=row,
            step=0,
            episode_done=False,
        )
        return self.state

    def _good_actions(self, diagnosis: str) -> List[str]:
        return DISEASE_ACTION_MAP.get(diagnosis, ["monitor"])

    def _compute_reward(self, s: PatientState, action: str) -> float:
        reward = 0.0
        true_disease = s.patient_row["disease_name"]
        true_severity = float(s.patient_row["disease_severity"])

        if s.diagnosis == true_disease:
            reward += 1.5
        else:
            reward -= 0.5

        if action in self._good_actions(true_disease):
            reward += 2.0
        elif action == "monitor" and true_severity < 0.30:
            reward += 0.5
        elif action == "discharge" and true_severity < 0.20:
            reward += 1.5
        elif action == "admit_hospital" and true_severity > 0.70:
            reward += 1.5
        else:
            reward -= 0.75

        if action in CONTRAINDICATIONS.get(true_disease, []):
            reward -= 3.0

        reward -= 0.1
        return float(reward)

    def _transition(self, s: PatientState, action: str) -> PatientState:
        new_features = s.features.copy()
        new_severity = s.severity
        true_disease = s.patient_row["disease_name"]

        if action in self._good_actions(true_disease):
            new_severity = max(0.0, new_severity - float(self.rng.uniform(0.05, 0.15)))
            new_features = np.clip(new_features + self.rng.normal(0, 0.01, size=new_features.shape), 0, 1)
        else:
            new_severity = min(1.0, new_severity + float(self.rng.uniform(0.01, 0.08)))

        new_step = s.step + 1
        done = (new_severity < 0.10) or (new_step >= self.MAX_STEPS) or (action == "discharge" and new_severity < 0.25)
        return PatientState(
            features=new_features.astype(np.float32),
            diagnosis=s.diagnosis,
            severity=float(new_severity),
            patient_row=s.patient_row,
            step=new_step,
            episode_done=done,
        )

    def step(self, action_idx: int) -> Tuple[PatientState, float, bool, dict]:
        assert self.state is not None, "Call reset() first"
        action = ACTIONS[action_idx]
        reward = self._compute_reward(self.state, action)
        next_state = self._transition(self.state, action)
        done = next_state.episode_done
        info = {
            "predicted_diagnosis": self.state.diagnosis,
            "true_diagnosis": self.state.patient_row["disease_name"],
            "severity": self.state.severity,
            "action": action,
        }
        self.state = next_state
        return next_state, reward, done, info


# ============================================================
# RL AGENTS (kept aligned with your earlier code structure)
# ============================================================
class TabularAgent:
    def __init__(self, n_actions: int, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state_key: tuple, greedy=False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state_key]))

    def decay_epsilon(self, decay=0.995, min_eps=0.01):
        self.epsilon = max(min_eps, self.epsilon * decay)


class SARSAAgent(TabularAgent):
    name = "SARSA"

    def update(self, s, a, r, s_next, a_next, done):
        td_target = r + (0 if done else self.gamma * self.Q[s_next][a_next])
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])


class QLearningAgent(TabularAgent):
    name = "Q-Learning"

    def update(self, s, a, r, s_next, done):
        td_target = r + (0 if done else self.gamma * np.max(self.Q[s_next]))
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])


class MonteCarloAgent(TabularAgent):
    name = "MonteCarlo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns = defaultdict(list)
        self.episode_buffer = []

    def store(self, state_key, action, reward):
        self.episode_buffer.append((state_key, action, reward))

    def finish_episode(self):
        G = 0.0
        visited = set()
        for (s, a, r) in reversed(self.episode_buffer):
            G = r + self.gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                self.returns[(s, a)].append(G)
                self.Q[s][a] = float(np.mean(self.returns[(s, a)]))
        self.episode_buffer.clear()


class MLP:
    def __init__(self, layer_sizes: List[int], lr=1e-3):
        self.lr = lr
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def predict(self, x: np.ndarray) -> np.ndarray:
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ W + b)
        return x @ self.weights[-1] + self.biases[-1]

    def copy_weights_from(self, other: "MLP"):
        self.weights = [W.copy() for W in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def update_output_weights(self, x: np.ndarray, target: float, idx: int):
        q_pred = self.predict(x)
        error = q_pred.copy()
        error[idx] -= target
        hidden = x.copy()
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.maximum(0, hidden @ W + b)
        self.weights[-1] -= self.lr * np.outer(hidden, error)
        self.biases[-1] -= self.lr * error


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    name = "DoubleDQN"

    def __init__(self, state_dim: int, n_actions: int,
                 hidden=[64, 64], lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.997, min_eps=0.05,
                 batch_size=32, target_update=50):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_eps = min_eps
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        sizes = [state_dim] + hidden + [n_actions]
        self.online = MLP(sizes, lr)
        self.target = MLP(sizes, lr)
        self.target.copy_weights_from(self.online)
        self.buffer = ReplayBuffer()

    def choose_action(self, state_vec: np.ndarray, greedy=False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q = self.online.predict(state_vec)
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        for s, a, r, s_next, done in batch:
            if done:
                target_q = r
            else:
                best_a = int(np.argmax(self.online.predict(s_next)))
                target_q = r + self.gamma * self.target.predict(s_next)[best_a]
            self.online.update_output_weights(s, target_q, a)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.copy_weights_from(self.online)

    def decay_epsilon(self):
        self.epsilon = max(self.min_eps, self.epsilon * self.epsilon_decay)


# ============================================================
# SAFETY + EXPLAINABILITY + IRL
# ============================================================
class SafetyLayer:
    def filter(self, recommended_action: str, patient: PatientState) -> Tuple[str, str]:
        true_disease = patient.patient_row["disease_name"]
        if recommended_action in CONTRAINDICATIONS.get(true_disease, []):
            return "monitor", f"[SAFETY] '{recommended_action}' contraindicated for {true_disease}."
        if recommended_action == "discharge" and patient.severity > 0.30:
            return "monitor", f"[SAFETY] Severity {patient.severity:.2f} too high for discharge."
        return recommended_action, "[OK] Action passed safety checks."


class Explainability:
    @staticmethod
    def explain(state: PatientState, action: str, q_values: Optional[np.ndarray] = None) -> str:
        lines = [
            f"Predicted Diagnosis : {state.diagnosis}",
            f"True Diagnosis      : {state.patient_row['disease_name']}",
            f"Severity            : {state.severity:.2f}",
            f"Recommended Action  : {action}",
        ]
        if q_values is not None:
            ranked = sorted(enumerate(q_values), key=lambda x: -x[1])[:3]
            lines.append("Top-3 Actions:")
            for rank, (idx, qv) in enumerate(ranked, 1):
                lines.append(f"  {rank}. {ACTIONS[idx]:24s} Q={qv:.4f}")
        return "\n".join(lines)


class MaxEntropyIRL:
    def __init__(self, feat_dim: int, n_actions: int, lr=0.01):
        self.w = np.zeros(feat_dim + 1)
        self.lr = lr

    def _phi(self, state_vec: np.ndarray, action_idx: int) -> np.ndarray:
        return np.append(state_vec, action_idx / len(ACTIONS))

    def reward(self, state_vec: np.ndarray, action_idx: int) -> float:
        return float(self.w @ self._phi(state_vec, action_idx))

    def update(self, expert_trajs, policy_trajs):
        expert_feat = np.mean([self._phi(s, a) for traj in expert_trajs for s, a in traj], axis=0)
        policy_feat = np.mean([self._phi(s, a) for traj in policy_trajs for s, a in traj], axis=0)
        self.w += self.lr * (expert_feat - policy_feat)


class ClinicalDecisionEngine:
    def __init__(self, dqn_agent: DQNAgent, irl_model: MaxEntropyIRL, safety: SafetyLayer):
        self.dqn = dqn_agent
        self.irl = irl_model
        self.safety = safety

    def recommend(self, patient: PatientState) -> Dict:
        sv = patient.to_vector()
        q_vals = self.dqn.online.predict(sv)
        for i in range(len(ACTIONS)):
            q_vals[i] += 0.3 * self.irl.reward(sv, i)
        ranked = np.argsort(q_vals)[::-1]
        best_action = ACTIONS[ranked[0]]
        safe_action, safety_msg = self.safety.filter(best_action, patient)
        explanation = Explainability.explain(patient, safe_action, q_vals)
        alternatives = [ACTIONS[i] for i in ranked[1:4]]
        return {
            "recommended_action": safe_action,
            "safety_message": safety_msg,
            "alternatives": alternatives,
            "explanation": explanation,
        }


# ============================================================
# TRAIN / EVAL UTILITIES
# ============================================================
def train_classical(agent, env, n_episodes=300, verbose=True):
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        sk = s.discretize()
        a = agent.choose_action(sk)
        done = False
        ep_r = 0.0
        while not done:
            s_next, r, done, _ = env.step(a)
            sk_next = s_next.discretize()
            a_next = agent.choose_action(sk_next)
            if isinstance(agent, SARSAAgent):
                agent.update(sk, a, r, sk_next, a_next, done)
            elif isinstance(agent, QLearningAgent):
                agent.update(sk, a, r, sk_next, done)
            elif isinstance(agent, MonteCarloAgent):
                agent.store(sk, a, r)
            sk, a = sk_next, a_next
            ep_r += r
        if isinstance(agent, MonteCarloAgent):
            agent.finish_episode()
        agent.decay_epsilon()
        rewards.append(ep_r)
        if verbose and (ep + 1) % 100 == 0:
            print(f"[{agent.name}] Episode {ep+1:4d} | Avg Reward (last 100): {np.mean(rewards[-100:]):.3f}")
    return rewards


def train_dqn(agent, env, n_episodes=300, verbose=True):
    rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            sv = s.to_vector()
            a = agent.choose_action(sv)
            s_next, r, done, _ = env.step(a)
            agent.store(sv, a, r, s_next.to_vector(), done)
            agent.train_step()
            s = s_next
            ep_r += r
        agent.decay_epsilon()
        rewards.append(ep_r)
        if verbose and (ep + 1) % 100 == 0:
            print(f"[DQN] Episode {ep+1:4d} | Avg Reward (last 100): {np.mean(rewards[-100:]):.3f}")
    return rewards


def evaluate_agent(agent, env, n_episodes=100, safety=None, tabular=True):
    rewards, severities, steps_list = [], [], []
    unsafe_count, success_count = 0, 0
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            state_repr = s.discretize() if tabular else s.to_vector()
            a = agent.choose_action(state_repr, greedy=True)
            action_name = ACTIONS[a]
            if safety is not None:
                action_name, msg = safety.filter(action_name, s)
                if "SAFETY" in msg:
                    unsafe_count += 1
                    a = ACTIONS.index(action_name)
            s, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
        severities.append(s.severity)
        steps_list.append(s.step)
        if s.severity < 0.25:
            success_count += 1
    total_steps = max(1, sum(steps_list))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_final_severity": float(np.mean(severities)),
        "mean_steps": float(np.mean(steps_list)),
        "unsafe_action_rate": float(unsafe_count / max(1, n_episodes * 5)),
        "safe_action_rate": float(1.0 - (unsafe_count / max(1, n_episodes * 5))),
        "treatment_success_rate": float(success_count / n_episodes),
    }


def collect_expert_trajectories(env, n=30):
    trajs = []
    for _ in range(n):
        s = env.reset()
        traj = []
        done = False
        while not done:
            good_actions = DISEASE_ACTION_MAP.get(s.patient_row["disease_name"], ["monitor"])
            action_name = random.choice(good_actions)
            a = ACTIONS.index(action_name)
            traj.append((s.to_vector(), a))
            s, _, done, _ = env.step(a)
        trajs.append(traj)
    return trajs


def train_irl(irl, env, expert_trajs, n_iter=15):
    for _ in range(n_iter):
        policy_trajs = []
        for _ in range(10):
            s = env.reset()
            traj = []
            done = False
            while not done:
                a = random.randrange(len(ACTIONS))
                traj.append((s.to_vector(), a))
                s, _, done, _ = env.step(a)
            policy_trajs.append(traj)
        irl.update(expert_trajs, policy_trajs)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 72)
    print("TWO-STAGE HEALTHCARE PIPELINE")
    print("Stage 1: CSV Diagnosis Training + Model Comparison")
    print("Stage 2: RL Treatment Learning")
    print("=" * 72)

    diagnosis_stage = DiagnosisStage(DATASET_PATH)
    df = diagnosis_stage.load()
    stage1 = diagnosis_stage.compare_models()

    print("\n[Stage 1] Model Comparison")
    print(stage1["comparison"].to_string(index=False))
    print(f"\nBest diagnosis model : {stage1['best_model_name']}")
    print(f"Diagnosis accuracy   : {stage1['best_accuracy']:.4f}")
    print(f"Severity MAE         : {stage1['severity_mae']:.4f}")

    env = DatasetHealthcareEnv(df, diagnosis_stage, seed=SEED)
    safety = SafetyLayer()
    state_dim = len(env.feature_cols_for_env) + 1

    print("\n[Stage 2] Training RL Agents")
    sarsa_agent = SARSAAgent(len(ACTIONS))
    qlearn_agent = QLearningAgent(len(ACTIONS))
    mc_agent = MonteCarloAgent(len(ACTIONS))
    dqn_agent = DQNAgent(state_dim, len(ACTIONS), hidden=[64, 64], epsilon=1.0)

    train_classical(sarsa_agent, env, n_episodes=N_RL_EPISODES, verbose=True)
    train_classical(qlearn_agent, env, n_episodes=N_RL_EPISODES, verbose=True)
    train_classical(mc_agent, env, n_episodes=N_RL_EPISODES, verbose=True)
    train_dqn(dqn_agent, env, n_episodes=N_RL_EPISODES, verbose=True)

    irl_model = MaxEntropyIRL(state_dim, len(ACTIONS))
    expert_trajs = collect_expert_trajectories(env, n=30)
    train_irl(irl_model, env, expert_trajs, n_iter=15)

    eval_results = {
        "SARSA": evaluate_agent(sarsa_agent, env, n_episodes=N_EVAL_EPISODES, safety=safety, tabular=True),
        "Q-Learning": evaluate_agent(qlearn_agent, env, n_episodes=N_EVAL_EPISODES, safety=safety, tabular=True),
        "MonteCarlo": evaluate_agent(mc_agent, env, n_episodes=N_EVAL_EPISODES, safety=safety, tabular=True),
        "DoubleDQN": evaluate_agent(dqn_agent, env, n_episodes=N_EVAL_EPISODES, safety=safety, tabular=False),
    }

    rl_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={"index": "agent"})
    rl_df.to_csv(os.path.join(OUTPUT_DIR, "rl_agent_comparison.csv"), index=False)

    best_rl_row = rl_df.sort_values(["mean_reward", "safe_action_rate", "treatment_success_rate"], ascending=False).iloc[0]
    best_rl_name = best_rl_row["agent"]
    safe_action_rate = float(best_rl_row["safe_action_rate"])
    treatment_success_rate = float(best_rl_row["treatment_success_rate"])

    total_model_accuracy = build_total_model_accuracy(
        diagnosis_accuracy=float(stage1["best_accuracy"]),
        severity_mae=float(stage1["severity_mae"]),
        safe_action_rate=safe_action_rate,
        treatment_success_rate=treatment_success_rate,
    )

    summary = {
        "best_diagnosis_model": stage1["best_model_name"],
        "diagnosis_accuracy": float(stage1["best_accuracy"]),
        "severity_mae": float(stage1["severity_mae"]),
        "best_rl_agent": best_rl_name,
        "best_rl_mean_reward": float(best_rl_row["mean_reward"]),
        "safe_action_rate": safe_action_rate,
        "treatment_success_rate": treatment_success_rate,
        "total_model_accuracy": total_model_accuracy,
    }

    with open(os.path.join(OUTPUT_DIR, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[RL Evaluation]")
    print(rl_df.to_string(index=False))

    print("\n[Final Summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:.4f}")
        else:
            print(f"{k:28s}: {v}")

    engine = ClinicalDecisionEngine(dqn_agent, irl_model, safety)
    sample_state = env.reset()
    rec = engine.recommend(sample_state)

    with open(os.path.join(OUTPUT_DIR, "sample_recommendation.txt"), "w", encoding="utf-8") as f:
        f.write(rec["explanation"] + "\n")
        f.write(f"Safety: {rec['safety_message']}\n")
        f.write(f"Alternatives: {rec['alternatives']}\n")

    print("\n[Sample Recommendation]")
    print(rec["explanation"])
    print(f"Safety       : {rec['safety_message']}")
    print(f"Alternatives : {rec['alternatives']}")
    print(f"\nSaved outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
