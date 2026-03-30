"""
Patient Dataset Builder
=======================
Builds a richly annotated synthetic patient dataset from scratch, covering:
  - Demographics & vitals
  - Eating habits (diet quality, meal frequency, hydration, junk food, etc.)
  - Lifestyle factors (smoking, alcohol, exercise, sleep, stress, occupation)
  - Disease name, root causes, severity
  - Prevention strategies tailored to each patient profile
  - RL-ready feature vectors

Diseases covered (10):
  Type 2 Diabetes | Hypertension | Coronary Artery Disease | COPD |
  Chronic Kidney Disease | Obesity | Depression | Asthma |
  Liver Cirrhosis | Anemia
"""

import numpy as np
import random
import csv 
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 0. DISEASE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────

DISEASE_KB: Dict[str, Dict] = {
    "Type 2 Diabetes": {
        "icd10": "E11",
        "system": "Endocrine",
        "risk_factors": {
            "diet":      ["high_sugar", "processed_foods", "low_fiber"],
            "lifestyle": ["sedentary", "smoking", "high_stress"],
            "vitals":    {"glucose": ">126", "bmi": ">27"},
        },
        "causes": [
            "Chronic high sugar and refined carbohydrate intake leads to insulin resistance",
            "Sedentary lifestyle reduces glucose uptake by muscle cells",
            "Excess visceral fat disrupts insulin signaling pathways",
            "Genetic predisposition combined with poor dietary habits",
            "Chronic stress elevates cortisol, impairing glucose regulation",
        ],
        "symptoms": ["excessive_thirst", "frequent_urination", "fatigue",
                     "blurred_vision", "slow_healing_wounds"],
        "prevention": [
            "Reduce refined sugar and processed carbohydrate intake",
            "Adopt a high-fiber, low-glycemic-index diet (whole grains, legumes, vegetables)",
            "Engage in at least 150 minutes of moderate aerobic exercise weekly",
            "Maintain healthy body weight (BMI 18.5–24.9)",
            "Monitor blood glucose regularly if pre-diabetic",
            "Limit alcohol consumption; avoid sugary beverages",
            "Manage stress through mindfulness, yoga, or therapy",
        ],
        "medications": ["Metformin", "GLP-1 agonists", "SGLT2 inhibitors"],
        "severity_lab_key": "glucose",
    },

    "Hypertension": {
        "icd10": "I10",
        "system": "Cardiovascular",
        "risk_factors": {
            "diet":      ["high_sodium", "low_potassium", "excessive_alcohol"],
            "lifestyle": ["sedentary", "smoking", "high_stress", "obesity"],
            "vitals":    {"blood_pressure": ">140"},
        },
        "causes": [
            "Excessive sodium intake increases blood volume and vascular pressure",
            "Sedentary lifestyle leads to arterial stiffness",
            "Smoking damages arterial walls and promotes vasoconstriction",
            "Obesity increases cardiac output and peripheral resistance",
            "Chronic psychological stress activates the renin-angiotensin system",
            "Insufficient potassium intake impairs sodium excretion",
        ],
        "symptoms": ["headache", "dizziness", "shortness_of_breath",
                     "chest_pain", "nosebleed"],
        "prevention": [
            "Limit sodium intake to less than 2,300 mg per day (DASH diet)",
            "Increase potassium-rich foods: bananas, spinach, beans",
            "Exercise regularly — aim for 30 minutes of brisk walking daily",
            "Quit smoking; avoid secondhand smoke",
            "Moderate alcohol: max 1 drink/day for women, 2 for men",
            "Achieve and maintain a healthy body weight",
            "Practice stress-reduction techniques: meditation, deep breathing",
            "Get 7–9 hours of quality sleep per night",
        ],
        "medications": ["ACE inhibitors", "Beta-blockers", "Calcium channel blockers", "Diuretics"],
        "severity_lab_key": "blood_pressure",
    },

    "Coronary Artery Disease": {
        "icd10": "I25",
        "system": "Cardiovascular",
        "risk_factors": {
            "diet":      ["high_saturated_fat", "high_trans_fat", "low_omega3"],
            "lifestyle": ["smoking", "sedentary", "high_stress", "excessive_alcohol"],
            "vitals":    {"heart_rate": ">90", "blood_pressure": ">130"},
        },
        "causes": [
            "High LDL cholesterol from saturated/trans fat diet causes plaque buildup",
            "Smoking oxidizes LDL and promotes endothelial damage",
            "Chronic hypertension strains coronary artery walls",
            "Sedentary lifestyle leads to dyslipidemia and obesity",
            "Diabetes accelerates atherosclerosis",
            "Chronic inflammation from poor diet damages arterial lining",
        ],
        "symptoms": ["chest_pain_angina", "shortness_of_breath",
                     "heart_palpitations", "fatigue", "jaw_or_arm_pain"],
        "prevention": [
            "Follow a heart-healthy diet: Mediterranean or DASH style",
            "Eliminate trans fats; limit saturated fat to <7% of daily calories",
            "Increase omega-3 intake: fatty fish (salmon, mackerel) twice weekly",
            "Quit smoking — risk halves within one year of cessation",
            "Exercise 150+ minutes per week, including cardio and strength training",
            "Control blood pressure, blood sugar, and cholesterol levels",
            "Reduce chronic stress with structured relaxation techniques",
            "Annual cardiac screenings after age 40 or if family history exists",
        ],
        "medications": ["Statins", "Aspirin", "Beta-blockers", "Nitrates"],
        "severity_lab_key": "heart_rate",
    },

    "COPD": {
        "icd10": "J44",
        "system": "Respiratory",
        "risk_factors": {
            "diet":      ["low_antioxidants", "low_vitamin_c", "processed_foods"],
            "lifestyle": ["smoking", "air_pollution_exposure", "occupational_dust"],
            "vitals":    {"oxygen_saturation": "<94"},
        },
        "causes": [
            "Long-term cigarette smoking destroys alveolar walls",
            "Occupational exposure to dust, chemicals, and fumes",
            "Chronic respiratory infections over years",
            "Alpha-1 antitrypsin genetic deficiency",
            "Indoor air pollution from biomass fuel cooking",
            "Low antioxidant intake accelerates oxidative lung damage",
        ],
        "symptoms": ["chronic_cough", "wheezing", "dyspnea",
                     "excess_mucus_production", "frequent_infections"],
        "prevention": [
            "Quit smoking immediately — the single most effective prevention",
            "Avoid occupational exposure: use appropriate respiratory protective equipment",
            "Reduce indoor air pollution: improve ventilation, avoid biomass burning",
            "Get vaccinated: annual flu and pneumococcal vaccines",
            "Eat antioxidant-rich foods: berries, citrus, leafy greens",
            "Practice breathing exercises (pursed-lip, diaphragmatic breathing)",
            "Avoid outdoor air pollution during high-alert days",
        ],
        "medications": ["Bronchodilators", "Inhaled corticosteroids", "Phosphodiesterase inhibitors"],
        "severity_lab_key": "oxygen_saturation",
    },

    "Chronic Kidney Disease": {
        "icd10": "N18",
        "system": "Renal",
        "risk_factors": {
            "diet":      ["high_protein", "high_sodium", "high_phosphorus", "low_hydration"],
            "lifestyle": ["smoking", "nsaid_overuse", "uncontrolled_diabetes"],
            "vitals":    {"creatinine": ">1.3"},
        },
        "causes": [
            "Long-standing uncontrolled diabetes damages glomeruli",
            "Chronic hypertension reduces renal perfusion",
            "Excessive NSAID/analgesic use causes nephrotoxicity",
            "Recurrent urinary tract infections lead to scarring",
            "Smoking reduces blood flow to the kidneys",
            "High protein diets increase filtration load on damaged kidneys",
        ],
        "symptoms": ["fatigue", "swelling_ankles", "frequent_urination_at_night",
                     "nausea", "decreased_appetite", "high_blood_pressure"],
        "prevention": [
            "Control blood sugar and blood pressure strictly",
            "Stay well-hydrated: aim for 2–3 liters of water daily",
            "Limit sodium, phosphorus, and excess protein intake",
            "Avoid overuse of NSAIDs (ibuprofen, naproxen)",
            "Quit smoking to preserve renal blood flow",
            "Annual kidney function tests (eGFR, creatinine) if diabetic or hypertensive",
            "Maintain healthy weight; follow renal-friendly diet if at risk",
        ],
        "medications": ["ACE inhibitors", "ARBs", "Phosphate binders", "Erythropoietin"],
        "severity_lab_key": "creatinine",
    },

    "Obesity": {
        "icd10": "E66",
        "system": "Metabolic",
        "risk_factors": {
            "diet":      ["high_caloric_intake", "processed_foods", "sugary_drinks",
                          "low_fiber", "frequent_fast_food"],
            "lifestyle": ["sedentary", "poor_sleep", "high_stress", "emotional_eating"],
            "vitals":    {"bmi": ">30"},
        },
        "causes": [
            "Caloric surplus from high-fat, high-sugar, ultra-processed food consumption",
            "Sedentary behavior reduces total energy expenditure",
            "Poor sleep disrupts ghrelin/leptin balance, increasing hunger",
            "Emotional eating in response to stress or depression",
            "Gut microbiome imbalance affects energy metabolism",
            "Genetic predisposition to fat storage",
        ],
        "symptoms": ["excess_body_fat", "fatigue", "joint_pain",
                     "sleep_apnea", "breathlessness_on_exertion"],
        "prevention": [
            "Follow a calorie-controlled, nutrient-dense diet",
            "Eat more vegetables, whole grains, lean proteins, and healthy fats",
            "Eliminate sugary beverages: sodas, juices, energy drinks",
            "Limit ultra-processed and fast foods",
            "Exercise at least 300 minutes per week for weight management",
            "Prioritize 7–9 hours of sleep to regulate appetite hormones",
            "Seek behavioral therapy for emotional or stress-driven eating",
            "Practice mindful eating: eat slowly, avoid screens during meals",
        ],
        "medications": ["Orlistat", "GLP-1 agonists (Semaglutide)", "Phentermine"],
        "severity_lab_key": "bmi",
    },

    "Depression": {
        "icd10": "F32",
        "system": "Mental Health",
        "risk_factors": {
            "diet":      ["low_omega3", "low_folate", "high_processed_food",
                          "low_vitamin_d", "high_sugar"],
            "lifestyle": ["social_isolation", "sedentary", "poor_sleep",
                          "high_stress", "substance_abuse"],
            "vitals":    {},
        },
        "causes": [
            "Chronic stress and trauma dysregulate the HPA axis",
            "Neuroinflammation from poor diet and gut dysbiosis",
            "Social isolation disrupts oxytocin and serotonin systems",
            "Low omega-3 fatty acids impair neuronal membrane function",
            "Vitamin D deficiency linked to serotonin synthesis disruption",
            "Sleep deprivation affects mood-regulating neurotransmitters",
        ],
        "symptoms": ["persistent_sadness", "loss_of_interest", "fatigue",
                     "sleep_disturbance", "appetite_change", "concentration_difficulty"],
        "prevention": [
            "Eat a brain-healthy diet: Mediterranean-style with omega-3s, B vitamins",
            "Exercise regularly — shown to be as effective as antidepressants for mild-moderate depression",
            "Maintain strong social connections; address loneliness proactively",
            "Prioritize 7–9 hours of restorative sleep",
            "Reduce alcohol and avoid recreational drugs",
            "Practice mindfulness, journaling, or cognitive behavioral techniques",
            "Get adequate sunlight exposure or supplement vitamin D",
            "Seek early professional mental health support when symptoms arise",
        ],
        "medications": ["SSRIs", "SNRIs", "Bupropion", "Psychotherapy (CBT)"],
        "severity_lab_key": "comorbidity_score",
    },

    "Asthma": {
        "icd10": "J45",
        "system": "Respiratory",
        "risk_factors": {
            "diet":      ["low_antioxidants", "high_processed_food", "food_allergies"],
            "lifestyle": ["smoking", "allergen_exposure", "air_pollution", "sedentary"],
            "vitals":    {"oxygen_saturation": "<95"},
        },
        "causes": [
            "Airway hypersensitivity triggered by allergens (dust, pollen, mold)",
            "Secondhand smoke exposure, especially in childhood",
            "Air pollution: fine particulate matter (PM2.5) inflames airways",
            "Respiratory viral infections trigger airway remodeling",
            "Occupational allergens (flour, isocyanates, latex)",
            "Obesity contributes to airway inflammation",
        ],
        "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness",
                     "nocturnal_cough", "exercise_induced_bronchospasm"],
        "prevention": [
            "Identify and avoid personal asthma triggers",
            "Keep home free of dust mites: use allergen-proof covers, vacuum regularly",
            "Avoid smoking and secondhand smoke exposure",
            "Monitor air quality index; stay indoors on high-pollution days",
            "Take prescribed preventive inhalers consistently",
            "Eat anti-inflammatory foods: fruits, vegetables, omega-3-rich fish",
            "Maintain healthy weight to reduce airway inflammation",
            "Get annual flu vaccination to prevent respiratory infections",
        ],
        "medications": ["ICS (Budesonide)", "LABA (Salmeterol)", "SABA rescue inhalers", "Montelukast"],
        "severity_lab_key": "oxygen_saturation",
    },

    "Liver Cirrhosis": {
        "icd10": "K74",
        "system": "Hepatic",
        "risk_factors": {
            "diet":      ["excessive_alcohol", "high_fat", "high_fructose"],
            "lifestyle": ["alcohol_abuse", "hepatitis_exposure", "sedentary"],
            "vitals":    {"wbc": ">11"},
        },
        "causes": [
            "Chronic alcohol abuse causes hepatic inflammation and fibrosis",
            "Non-alcoholic fatty liver disease (NAFLD) from obesity and insulin resistance",
            "Chronic hepatitis B or C viral infection",
            "Autoimmune hepatitis leads to progressive scarring",
            "Long-term bile duct disease disrupts hepatocyte function",
            "High fructose intake promotes de novo lipogenesis and hepatic fat",
        ],
        "symptoms": ["jaundice", "abdominal_swelling", "fatigue",
                     "easy_bruising", "spider_angiomas", "mental_confusion"],
        "prevention": [
            "Abstain from or strictly limit alcohol consumption",
            "Get vaccinated for Hepatitis A and B",
            "Maintain healthy body weight to prevent NAFLD",
            "Eat a liver-friendly diet: low saturated fat, low fructose",
            "Avoid sharing needles or unprotected exposure to blood products",
            "Use medications carefully — avoid hepatotoxic drugs without medical advice",
            "Get regular liver function tests if you have risk factors",
            "Treat diabetes and obesity to prevent NAFLD progression",
        ],
        "medications": ["Diuretics (Spironolactone)", "Beta-blockers", "Lactulose", "Antiviral agents"],
        "severity_lab_key": "wbc",
    },

    "Anemia": {
        "icd10": "D64",
        "system": "Hematologic",
        "risk_factors": {
            "diet":      ["low_iron", "low_vitamin_b12", "low_folate",
                          "vegetarian_without_supplementation"],
            "lifestyle": ["heavy_menstruation", "chronic_disease", "poor_nutrition"],
            "vitals":    {"hemoglobin": "<12"},
        },
        "causes": [
            "Inadequate dietary iron intake, especially in vegetarians/vegans",
            "Vitamin B12 deficiency (pernicious anemia or vegan diet without supplementation)",
            "Chronic blood loss: heavy menstruation, GI bleeding",
            "Bone marrow disorders reducing red blood cell production",
            "Chronic diseases (CKD, rheumatoid arthritis) suppress erythropoietin",
            "Folate deficiency from poor diet or malabsorption",
        ],
        "symptoms": ["fatigue", "pallor", "dizziness", "shortness_of_breath",
                     "cold_extremities", "brittle_nails", "rapid_heartbeat"],
        "prevention": [
            "Eat iron-rich foods: red meat, poultry, fish, legumes, fortified cereals",
            "Pair iron sources with vitamin C to enhance absorption",
            "Supplement B12 if vegetarian/vegan",
            "Ensure adequate folate intake: leafy greens, lentils, beans",
            "Investigate and treat any source of chronic blood loss",
            "Regular CBC screening if at risk (women of reproductive age, elderly)",
            "Avoid tea/coffee immediately after iron-rich meals (inhibit absorption)",
        ],
        "medications": ["Ferrous sulfate (oral iron)", "B12 injections", "Folic acid", "Erythropoietin"],
        "severity_lab_key": "hemoglobin",
    },
}

DISEASE_NAMES = list(DISEASE_KB.keys())

# ─────────────────────────────────────────────────────────────
# 1. CATEGORICAL OPTION POOLS
# ─────────────────────────────────────────────────────────────

DIET_TYPES = [
    "Balanced (mixed whole foods)",
    "Mediterranean (olive oil, fish, vegetables)",
    "Vegetarian (plant-based, dairy/eggs allowed)",
    "Vegan (fully plant-based)",
    "Keto (very low carb, high fat)",
    "High-sugar / high-processed",
    "Fast-food dominant",
    "High-sodium / fried foods",
    "Low-calorie restrictive",
    "Traditional regional diet",
]

MEAL_FREQUENCIES = [
    "1 meal/day (OMAD)",
    "2 meals/day",
    "3 meals/day (standard)",
    "4–5 small meals/day (frequent)",
    "Irregular / skips meals often",
]

HYDRATION_LEVELS = [
    "Very low (<1L/day)",
    "Low (1–1.5L/day)",
    "Adequate (1.5–2.5L/day)",
    "Good (2.5–3.5L/day)",
    "Excellent (>3.5L/day)",
]

JUNK_FOOD_FREQUENCY = [
    "Never",
    "Rarely (once a month)",
    "Occasionally (1–2x/week)",
    "Frequently (3–5x/week)",
    "Daily",
]

ALCOHOL_LEVELS = [
    "None",
    "Light (1–2 drinks/week)",
    "Moderate (3–7 drinks/week)",
    "Heavy (>7 drinks/week)",
    "Binge drinking",
]

SMOKING_STATUS = [
    "Never smoked",
    "Former smoker (quit >5 years ago)",
    "Former smoker (quit 1–5 years ago)",
    "Current light smoker (<10 cigs/day)",
    "Current heavy smoker (>10 cigs/day)",
]

EXERCISE_LEVELS = [
    "Sedentary (no exercise)",
    "Light (walking <30 min/day)",
    "Moderate (150 min/week aerobic)",
    "Active (300+ min/week mixed)",
    "Athlete (daily intense training)",
]

SLEEP_QUALITY = [
    "Very poor (<5 hrs, frequent waking)",
    "Poor (5–6 hrs, unrefreshing)",
    "Fair (6–7 hrs, occasional disruption)",
    "Good (7–8 hrs, mostly restful)",
    "Excellent (8–9 hrs, deep restful sleep)",
]

STRESS_LEVELS = [
    "Minimal (relaxed lifestyle)",
    "Low (manageable daily stressors)",
    "Moderate (work/family pressures)",
    "High (chronic workplace/financial stress)",
    "Severe (burnout, traumatic events)",
]

OCCUPATIONS = [
    "Office/desk worker",
    "Manual laborer (construction, farming)",
    "Healthcare professional",
    "Teacher / academic",
    "Business / management",
    "Homemaker",
    "Student",
    "Retired",
    "Unemployed",
    "Driver / transport worker",
    "Factory / industrial worker",
    "IT / technology worker",
]

GENDERS = ["Male", "Female", "Non-binary"]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

FAMILY_HISTORY_CONDITIONS = [
    "None",
    "Diabetes",
    "Heart disease",
    "Hypertension",
    "Cancer",
    "Kidney disease",
    "Mental illness",
    "Obesity",
    "Multiple conditions",
]

# ─────────────────────────────────────────────────────────────
# 2. PATIENT RECORD DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class PatientRecord:
    # Identity
    patient_id:        str
    age:               int
    gender:            str
    blood_group:       str
    height_cm:         float
    weight_kg:         float
    bmi:               float
    occupation:        str

    # Eating habits
    diet_type:         str
    meal_frequency:    str
    hydration:         str
    junk_food_freq:    str
    fruit_veg_servings_per_day: float   # 0–10
    sugar_intake_g_per_day:     float   # 0–200
    sodium_intake_mg_per_day:   float   # 500–5000

    # Lifestyle
    smoking_status:    str
    alcohol_level:     str
    exercise_level:    str
    sleep_quality:     str
    stress_level:      str
    screen_time_hours: float            # daily screen time
    outdoor_time_hours:float            # daily outdoor activity
    family_history:    str

    # Clinical vitals (at assessment)
    heart_rate:        float
    blood_pressure_systolic: float
    oxygen_saturation: float
    temperature_c:     float
    glucose_mg_dl:     float

    # Lab results
    wbc:               float
    creatinine:        float
    hemoglobin:        float
    sodium_meq:        float
    potassium:         float

    # Disease information
    disease_name:      str
    icd10_code:        str
    body_system:       str
    disease_severity:  float            # 0.0 mild → 1.0 critical
    primary_causes:    List[str]
    symptoms_present:  List[str]
    prevention_plan:   List[str]
    recommended_medications: List[str]

    # RL feature vector (auto-computed)
    feature_vector:    List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert lists to pipe-separated strings for CSV
        for k in ["primary_causes", "symptoms_present",
                  "prevention_plan", "recommended_medications"]:
            d[k] = " | ".join(d[k])
        d["feature_vector"] = json.dumps(d["feature_vector"])
        return d


# ─────────────────────────────────────────────────────────────
# 3. RISK-WEIGHTED DISEASE ASSIGNMENT
# ─────────────────────────────────────────────────────────────

def compute_disease_risk_scores(record_partial: dict) -> Dict[str, float]:
    """
    Score each disease based on the patient's lifestyle & vitals.
    Returns a probability-like score per disease.
    """
    scores = {d: 1.0 for d in DISEASE_NAMES}
    age    = record_partial["age"]
    bmi    = record_partial["bmi"]

    # ── Type 2 Diabetes ──────────────────────────────────────
    if record_partial["diet_type"] in ["High-sugar / high-processed", "Fast-food dominant"]:
        scores["Type 2 Diabetes"] += 3
    if bmi > 27:
        scores["Type 2 Diabetes"] += 2
    if record_partial["exercise_level"] == "Sedentary (no exercise)":
        scores["Type 2 Diabetes"] += 2
    if record_partial["glucose_mg_dl"] > 110:
        scores["Type 2 Diabetes"] += 3
    if "Diabetes" in record_partial["family_history"]:
        scores["Type 2 Diabetes"] += 2
    if record_partial["stress_level"] in ["High", "Severe"]:
        scores["Type 2 Diabetes"] += 1

    # ── Hypertension ─────────────────────────────────────────
    if record_partial["sodium_intake_mg_per_day"] > 3000:
        scores["Hypertension"] += 3
    if record_partial["blood_pressure_systolic"] > 130:
        scores["Hypertension"] += 4
    if record_partial["smoking_status"] in ["Current light smoker (<10 cigs/day)",
                                             "Current heavy smoker (>10 cigs/day)"]:
        scores["Hypertension"] += 2
    if bmi > 28:
        scores["Hypertension"] += 2
    if "Hypertension" in record_partial["family_history"]:
        scores["Hypertension"] += 2
    if record_partial["alcohol_level"] in ["Heavy (>7 drinks/week)", "Binge drinking"]:
        scores["Hypertension"] += 2

    # ── Coronary Artery Disease ───────────────────────────────
    if record_partial["smoking_status"] in ["Current heavy smoker (>10 cigs/day)"]:
        scores["Coronary Artery Disease"] += 4
    if record_partial["blood_pressure_systolic"] > 140:
        scores["Coronary Artery Disease"] += 3
    if age > 50:
        scores["Coronary Artery Disease"] += 2
    if record_partial["diet_type"] in ["High-sodium / fried foods", "Fast-food dominant",
                                        "Keto (very low carb, high fat)"]:
        scores["Coronary Artery Disease"] += 2
    if "Heart disease" in record_partial["family_history"]:
        scores["Coronary Artery Disease"] += 3

    # ── COPD ──────────────────────────────────────────────────
    if "heavy smoker" in record_partial["smoking_status"]:
        scores["COPD"] += 5
    if record_partial["oxygen_saturation"] < 94:
        scores["COPD"] += 4
    if record_partial["occupation"] in ["Manual laborer (construction, farming)",
                                         "Factory / industrial worker"]:
        scores["COPD"] += 2
    if age > 45:
        scores["COPD"] += 1

    # ── Chronic Kidney Disease ────────────────────────────────
    if record_partial["creatinine"] > 1.3:
        scores["Chronic Kidney Disease"] += 5
    if record_partial["hydration"] in ["Very low (<1L/day)", "Low (1–1.5L/day)"]:
        scores["Chronic Kidney Disease"] += 2
    if "Kidney disease" in record_partial["family_history"]:
        scores["Chronic Kidney Disease"] += 3
    if record_partial["glucose_mg_dl"] > 126:
        scores["Chronic Kidney Disease"] += 2

    # ── Obesity ───────────────────────────────────────────────
    if bmi > 30:
        scores["Obesity"] += 6
    if record_partial["exercise_level"] == "Sedentary (no exercise)":
        scores["Obesity"] += 3
    if record_partial["diet_type"] in ["Fast-food dominant", "High-sugar / high-processed"]:
        scores["Obesity"] += 3
    if record_partial["sleep_quality"] in ["Very poor (<5 hrs, frequent waking)",
                                            "Poor (5–6 hrs, unrefreshing)"]:
        scores["Obesity"] += 1

    # ── Depression ────────────────────────────────────────────
    if record_partial["stress_level"] in ["High (chronic workplace/financial stress)",
                                           "Severe (burnout, traumatic events)"]:
        scores["Depression"] += 4
    if record_partial["sleep_quality"] in ["Very poor (<5 hrs, frequent waking)"]:
        scores["Depression"] += 3
    if record_partial["exercise_level"] == "Sedentary (no exercise)":
        scores["Depression"] += 2
    if "Mental illness" in record_partial["family_history"]:
        scores["Depression"] += 3
    if record_partial["alcohol_level"] in ["Heavy (>7 drinks/week)", "Binge drinking"]:
        scores["Depression"] += 2

    # ── Asthma ────────────────────────────────────────────────
    if record_partial["smoking_status"] != "Never smoked":
        scores["Asthma"] += 2
    if record_partial["oxygen_saturation"] < 95:
        scores["Asthma"] += 3
    if record_partial["outdoor_time_hours"] < 0.5:
        scores["Asthma"] += 1
    if age < 35:
        scores["Asthma"] += 1

    # ── Liver Cirrhosis ───────────────────────────────────────
    if record_partial["alcohol_level"] in ["Heavy (>7 drinks/week)", "Binge drinking"]:
        scores["Liver Cirrhosis"] += 6
    if record_partial["diet_type"] in ["High-sugar / high-processed", "Fast-food dominant",
                                        "High-sodium / fried foods"]:
        scores["Liver Cirrhosis"] += 2
    if bmi > 30:
        scores["Liver Cirrhosis"] += 2
    if record_partial["wbc"] > 11:
        scores["Liver Cirrhosis"] += 3

    # ── Anemia ────────────────────────────────────────────────
    if record_partial["hemoglobin"] < 12:
        scores["Anemia"] += 6
    if record_partial["diet_type"] in ["Vegan (fully plant-based)",
                                        "Vegetarian (plant-based, dairy/eggs allowed)",
                                        "Low-calorie restrictive"]:
        scores["Anemia"] += 3
    if record_partial["fruit_veg_servings_per_day"] < 2:
        scores["Anemia"] += 2
    if record_partial.get("gender") == "Female" and age < 50:
        scores["Anemia"] += 1

    return scores


def assign_disease(scores: Dict[str, float], rng: np.random.Generator) -> str:
    """Sample disease weighted by risk scores."""
    names  = list(scores.keys())
    values = np.array([scores[n] for n in names], dtype=float)
    probs  = values / values.sum()
    return rng.choice(names, p=probs)


# ─────────────────────────────────────────────────────────────
# 4. SEVERITY COMPUTATION
# ─────────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS = {
    "glucose_mg_dl":         {"mild": 126, "moderate": 180, "severe": 250},
    "blood_pressure_systolic":{"mild": 130, "moderate": 160, "severe": 180},
    "heart_rate":             {"mild": 100, "moderate": 115, "severe": 130},
    "oxygen_saturation":      {"mild": 94,  "moderate": 90,  "severe": 85},   # lower = worse
    "creatinine":             {"mild": 1.3, "moderate": 2.0, "severe": 4.0},
    "bmi":                    {"mild": 30,  "moderate": 35,  "severe": 40},
    "hemoglobin":             {"mild": 11,  "moderate": 9,   "severe": 7},    # lower = worse
    "wbc":                    {"mild": 11,  "moderate": 15,  "severe": 20},
}

def compute_severity(disease: str, record_partial: dict) -> float:
    """Return a severity score 0.1 (mild) – 0.9 (critical)."""
    key = DISEASE_KB[disease].get("severity_lab_key", "glucose_mg_dl")
    thresholds = SEVERITY_THRESHOLDS.get(key)
    if thresholds is None:
        return round(random.uniform(0.2, 0.6), 2)

    val = record_partial.get(key, 0)
    # For metrics where lower = worse
    lower_worse = key in ("oxygen_saturation", "hemoglobin")
    if lower_worse:
        if val <= thresholds["severe"]:   return round(random.uniform(0.75, 0.95), 2)
        if val <= thresholds["moderate"]: return round(random.uniform(0.45, 0.74), 2)
        if val <= thresholds["mild"]:     return round(random.uniform(0.20, 0.44), 2)
        return round(random.uniform(0.05, 0.19), 2)
    else:
        if val >= thresholds["severe"]:   return round(random.uniform(0.75, 0.95), 2)
        if val >= thresholds["moderate"]: return round(random.uniform(0.45, 0.74), 2)
        if val >= thresholds["mild"]:     return round(random.uniform(0.20, 0.44), 2)
        return round(random.uniform(0.05, 0.19), 2)


# ─────────────────────────────────────────────────────────────
# 5. FEATURE VECTOR BUILDER (for RL)
# ─────────────────────────────────────────────────────────────

def build_feature_vector(record: dict) -> List[float]:
    """
    Encode all patient fields into a normalized numeric vector for RL.
    """
    def safe_index(lst, val):
        try:   return lst.index(val) / max(len(lst) - 1, 1)
        except ValueError: return 0.0

    fv = [
        # Demographics
        record["age"] / 100.0,
        record["bmi"] / 50.0,
        safe_index(GENDERS, record["gender"]),

        # Diet & eating
        safe_index(DIET_TYPES,        record["diet_type"]),
        safe_index(MEAL_FREQUENCIES,  record["meal_frequency"]),
        safe_index(HYDRATION_LEVELS,  record["hydration"]),
        safe_index(JUNK_FOOD_FREQUENCY, record["junk_food_freq"]),
        record["fruit_veg_servings_per_day"] / 10.0,
        record["sugar_intake_g_per_day"] / 200.0,
        record["sodium_intake_mg_per_day"] / 5000.0,

        # Lifestyle
        safe_index(SMOKING_STATUS,      record["smoking_status"]),
        safe_index(ALCOHOL_LEVELS,      record["alcohol_level"]),
        safe_index(EXERCISE_LEVELS,     record["exercise_level"]),
        safe_index(SLEEP_QUALITY,       record["sleep_quality"]),
        safe_index(STRESS_LEVELS,       record["stress_level"]),
        record["screen_time_hours"] / 16.0,
        record["outdoor_time_hours"] / 8.0,
        safe_index(FAMILY_HISTORY_CONDITIONS, record["family_history"]),

        # Vitals (normalized)
        (record["heart_rate"] - 40) / 160.0,
        (record["blood_pressure_systolic"] - 60) / 140.0,
        (record["oxygen_saturation"] - 70) / 30.0,
        (record["temperature_c"] - 35) / 6.0,
        (record["glucose_mg_dl"] - 50) / 300.0,

        # Labs
        record["wbc"] / 30.0,
        record["creatinine"] / 10.0,
        record["hemoglobin"] / 20.0,
        (record["sodium_meq"] - 100) / 80.0,
        record["potassium"] / 8.0,

        # Disease encoding
        safe_index(DISEASE_NAMES, record["disease_name"]),
        record["disease_severity"],
    ]
    return [round(float(np.clip(v, 0, 1)), 4) for v in fv]


# ─────────────────────────────────────────────────────────────
# 6. PATIENT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_patient_record(patient_id: int,
                             rng: np.random.Generator) -> PatientRecord:
    """
    Generate one complete synthetic patient record.
    """
    # ── Demographics ─────────────────────────────────────────
    age     = int(rng.integers(18, 85))
    gender  = rng.choice(GENDERS)
    height  = round(float(rng.uniform(150, 195)), 1)
    weight  = round(float(rng.uniform(45, 140)), 1)
    bmi     = round(weight / ((height / 100) ** 2), 1)
    bgroup  = rng.choice(BLOOD_GROUPS)
    occup   = rng.choice(OCCUPATIONS)

    # ── Eating habits ─────────────────────────────────────────
    diet_type   = rng.choice(DIET_TYPES)
    meal_freq   = rng.choice(MEAL_FREQUENCIES)
    hydration   = rng.choice(HYDRATION_LEVELS)
    junk_freq   = rng.choice(JUNK_FOOD_FREQUENCY)
    fv_servings = round(float(rng.uniform(0, 9)), 1)
    sugar_g     = round(float(rng.uniform(10, 190)), 1)
    sodium_mg   = round(float(rng.uniform(600, 4800)), 0)

    # ── Lifestyle ─────────────────────────────────────────────
    smoking   = rng.choice(SMOKING_STATUS)
    alcohol   = rng.choice(ALCOHOL_LEVELS)
    exercise  = rng.choice(EXERCISE_LEVELS)
    sleep_q   = rng.choice(SLEEP_QUALITY)
    stress    = rng.choice(STRESS_LEVELS)
    screen_hr = round(float(rng.uniform(0.5, 14)), 1)
    outdoor_h = round(float(rng.uniform(0, 5)), 1)
    fam_hist  = rng.choice(FAMILY_HISTORY_CONDITIONS)

    # ── Vitals (partly correlated with lifestyle) ─────────────
    base_hr  = 75
    if exercise == "Athlete (daily intense training)": base_hr -= 15
    if exercise == "Sedentary (no exercise)":          base_hr += 10
    if stress in ["High (chronic workplace/financial stress)",
                  "Severe (burnout, traumatic events)"]:     base_hr += 8
    hr = round(float(np.clip(rng.normal(base_hr, 12), 40, 170)), 1)

    base_bp = 115
    if "heavy smoker" in smoking:   base_bp += 20
    if sodium_mg > 3500:            base_bp += 15
    if bmi > 30:                    base_bp += 10
    bp_sys = round(float(np.clip(rng.normal(base_bp, 18), 70, 220)), 1)

    spo2 = round(float(np.clip(rng.normal(97.5, 1.5), 80, 100)), 1)
    if "smoker" in smoking:         spo2 = round(float(np.clip(spo2 - rng.uniform(1, 5), 80, 100)), 1)

    temp = round(float(np.clip(rng.normal(36.8, 0.5), 35.0, 41.5)), 1)

    base_glucose = 85
    if diet_type in ["High-sugar / high-processed", "Fast-food dominant"]:
        base_glucose += 40
    if bmi > 30: base_glucose += 20
    glucose = round(float(np.clip(rng.normal(base_glucose, 25), 50, 400)), 1)

    # ── Labs ──────────────────────────────────────────────────
    wbc       = round(float(np.clip(rng.normal(7.5, 2.5), 1, 30)), 2)
    creat     = round(float(np.clip(rng.normal(1.0, 0.5), 0.3, 10)), 2)
    hgb       = round(float(np.clip(rng.normal(13.5, 1.8), 5, 20)), 1)
    sodium_eq = round(float(np.clip(rng.normal(140, 4), 115, 160)), 1)
    potassium = round(float(np.clip(rng.normal(4.1, 0.5), 2.5, 7.5)), 2)

    # ── Disease assignment ────────────────────────────────────
    partial = {
        "age": age, "bmi": bmi, "gender": gender,
        "occupation": occup,
        "diet_type": diet_type, "meal_frequency": meal_freq,
        "hydration": hydration, "junk_food_freq": junk_freq,
        "fruit_veg_servings_per_day": fv_servings,
        "sugar_intake_g_per_day": sugar_g,
        "sodium_intake_mg_per_day": sodium_mg,
        "smoking_status": smoking, "alcohol_level": alcohol,
        "exercise_level": exercise, "sleep_quality": sleep_q,
        "stress_level": stress, "screen_time_hours": screen_hr,
        "outdoor_time_hours": outdoor_h, "family_history": fam_hist,
        "heart_rate": hr, "blood_pressure_systolic": bp_sys,
        "oxygen_saturation": spo2, "temperature_c": temp,
        "glucose_mg_dl": glucose,
        "wbc": wbc, "creatinine": creat, "hemoglobin": hgb,
        "sodium_meq": sodium_eq, "potassium": potassium,
    }

    risk_scores  = compute_disease_risk_scores(partial)
    disease_name = assign_disease(risk_scores, rng)
    kb           = DISEASE_KB[disease_name]
    severity     = compute_severity(disease_name, partial)

    # Select 3–4 causes, 3–5 symptoms, all prevention tips
    n_causes   = rng.integers(3, min(5, len(kb["causes"])) + 1)
    causes     = list(rng.choice(kb["causes"], size=int(n_causes), replace=False))
    symptoms   = list(rng.choice(kb["symptoms"],
                                  size=min(int(rng.integers(3, 6)), len(kb["symptoms"])),
                                  replace=False))
    prevention = kb["prevention"]
    meds       = kb["medications"]

    partial["disease_name"]    = disease_name
    partial["disease_severity"] = severity
    fv = build_feature_vector(partial)

    return PatientRecord(
        patient_id   = f"PAT-{patient_id:05d}",
        age=age, gender=gender, blood_group=bgroup,
        height_cm=height, weight_kg=weight, bmi=bmi,
        occupation=occup,

        diet_type=diet_type, meal_frequency=meal_freq,
        hydration=hydration, junk_food_freq=junk_freq,
        fruit_veg_servings_per_day=fv_servings,
        sugar_intake_g_per_day=sugar_g,
        sodium_intake_mg_per_day=sodium_mg,

        smoking_status=smoking, alcohol_level=alcohol,
        exercise_level=exercise, sleep_quality=sleep_q,
        stress_level=stress, screen_time_hours=screen_hr,
        outdoor_time_hours=outdoor_h, family_history=fam_hist,

        heart_rate=hr, blood_pressure_systolic=bp_sys,
        oxygen_saturation=spo2, temperature_c=temp,
        glucose_mg_dl=glucose,
        wbc=wbc, creatinine=creat, hemoglobin=hgb,
        sodium_meq=sodium_eq, potassium=potassium,

        disease_name=disease_name,
        icd10_code=kb["icd10"],
        body_system=kb["system"],
        disease_severity=severity,
        primary_causes=causes,
        symptoms_present=symptoms,
        prevention_plan=prevention,
        recommended_medications=meds,
        feature_vector=fv,
    )


# ─────────────────────────────────────────────────────────────
# 7. DATASET BUILDER
# ─────────────────────────────────────────────────────────────

def build_dataset(n_patients: int = 500,
                  seed: int = 42,
                  output_dir: str = ".") -> List[PatientRecord]:
    """Generate n_patients records and save to CSV and JSON."""
    rng      = np.random.default_rng(seed)
    records  = []

    print(f"Generating {n_patients} patient records...")
    for i in range(1, n_patients + 1):
        rec = generate_patient_record(i, rng)
        records.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{n_patients} records generated")

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "patient_dataset.csv")
    dicts    = [r.to_dict() for r in records]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dicts[0].keys())
        writer.writeheader()
        writer.writerows(dicts)
    print(f"\nCSV saved  → {csv_path}  ({os.path.getsize(csv_path)//1024} KB)")

    # ── Save JSON (full structure) ─────────────────────────────
    json_path = os.path.join(output_dir, "patient_dataset.json")
    json_records = []
    for r in records:
        d = asdict(r)
        json_records.append(d)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)
    print(f"JSON saved → {json_path} ({os.path.getsize(json_path)//1024} KB)")

    return records


# ─────────────────────────────────────────────────────────────
# 8. DATASET STATISTICS & PREVIEW
# ─────────────────────────────────────────────────────────────

def print_dataset_stats(records: List[PatientRecord]):
    print("\n" + "=" * 65)
    print("  DATASET STATISTICS")
    print("=" * 65)

    # Disease distribution
    from collections import Counter
    disease_counts = Counter(r.disease_name for r in records)
    print(f"\n{'Disease':<32} {'Count':>6}  {'%':>6}")
    print("-" * 48)
    for name, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records)
        print(f"{name:<32} {count:>6}  {pct:>5.1f}%")

    # Severity distribution
    sevs = [r.disease_severity for r in records]
    print(f"\nSeverity  mean={np.mean(sevs):.3f}  "
          f"std={np.std(sevs):.3f}  "
          f"min={np.min(sevs):.3f}  "
          f"max={np.max(sevs):.3f}")

    # Feature vector dimension
    print(f"\nFeature vector dimension : {len(records[0].feature_vector)}")
    print(f"Total records            : {len(records)}")

    # Sample record
    r = records[0]
    print(f"\n{'─'*65}")
    print(f"SAMPLE RECORD  →  {r.patient_id}")
    print(f"{'─'*65}")
    print(f"  Age/Gender/BMI    : {r.age}y  {r.gender}  BMI={r.bmi}")
    print(f"  Occupation        : {r.occupation}")
    print(f"  Diet              : {r.diet_type}")
    print(f"  Meal frequency    : {r.meal_frequency}")
    print(f"  Hydration         : {r.hydration}")
    print(f"  Junk food         : {r.junk_food_freq}")
    print(f"  Fruit & veg       : {r.fruit_veg_servings_per_day} servings/day")
    print(f"  Sugar intake      : {r.sugar_intake_g_per_day} g/day")
    print(f"  Sodium intake     : {r.sodium_intake_mg_per_day} mg/day")
    print(f"  Smoking           : {r.smoking_status}")
    print(f"  Alcohol           : {r.alcohol_level}")
    print(f"  Exercise          : {r.exercise_level}")
    print(f"  Sleep quality     : {r.sleep_quality}")
    print(f"  Stress level      : {r.stress_level}")
    print(f"  Family history    : {r.family_history}")
    print(f"\n  Vitals:")
    print(f"    HR={r.heart_rate}  BP={r.blood_pressure_systolic}  "
          f"SpO2={r.oxygen_saturation}%  Temp={r.temperature_c}°C  "
          f"Glucose={r.glucose_mg_dl} mg/dL")
    print(f"\n  Labs:")
    print(f"    WBC={r.wbc}  Creatinine={r.creatinine}  "
          f"Hgb={r.hemoglobin}  Na={r.sodium_meq}  K={r.potassium}")
    print(f"\n  ── DISEASE PROFILE ──")
    print(f"  Disease Name      : {r.disease_name}  [{r.icd10_code}]")
    print(f"  Body System       : {r.body_system}")
    print(f"  Severity Score    : {r.disease_severity:.2f}")
    print(f"\n  Primary Causes:")
    for c in r.primary_causes:
        print(f"    • {c}")
    print(f"\n  Symptoms Present  : {', '.join(r.symptoms_present)}")
    print(f"\n  Prevention Plan:")
    for p in r.prevention_plan:
        print(f"    ✓ {p}")
    print(f"\n  Recommended Meds  : {', '.join(r.recommended_medications)}")
    print(f"  Feature Vector    : {r.feature_vector[:8]} ... "
          f"[{len(r.feature_vector)} dims]")


# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = r"C:\Users\saivi\healthcare rl\output"
    records    = build_dataset(n_patients=500, seed=42, output_dir=OUTPUT_DIR)
    print_dataset_stats(records)
