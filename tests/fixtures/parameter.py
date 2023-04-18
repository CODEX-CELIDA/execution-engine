criteria_defs = {
    "COVID19": {"type": "condition", "static": True},
    "VENOUS_THROMBOSIS": {"type": "condition", "static": True},
    "HIT2": {"type": "condition", "static": True},
    "HEPARIN_ALLERGY": {"type": "observation", "static": True},
    "HEPARINOID_ALLERGY": {"type": "observation", "static": True},
    "THROMBOCYTOPENIA": {"type": "condition", "static": True},
    # 'pulmonary_embolism': {'type': 'condition', 'static': True},
    # 'ards': {'type': 'condition', 'static': True},
    "WEIGHT": {"type": "observation", "static": True, "threshold": 70},
    "DALTEPARIN": {
        "type": "drug",
        "dosage_threshold": 5000,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "ENOXAPARIN": {
        "type": "drug",
        "dosage_threshold": 40,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "NADROPARIN_LOW_WEIGHT": {
        "type": "drug",
        "dosage_threshold": 3800,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "NADROPARIN_HIGH_WEIGHT": {
        "type": "drug",
        "dosage_threshold": 5700,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "CERTOPARIN": {
        "type": "drug",
        "dosage_threshold": 3000,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "FONDAPARINUX": {
        "type": "drug",
        "dosage": 2.5,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "drug_certoparin": {
        "type": "drug",
        "dosage": 1,
        "doses_per_day": [0, 5],
        "p_dose_specific_day": 0.2,
        "static": False,
    },
    "HEPARIN": {"type": "drug", "dosage": 1, "doses_per_day": [0, 5], "static": False},
    "AGARTROBAN": {
        "type": "drug",
        "dosage": 1,
        "doses_per_day": [0, 5],
        "static": False,
    },
    "icu": {
        "type": "episode",
        "range": [0, 30],
        "n_occurrences": [0, 2],
        "static": False,
    },
    "pronining": {
        "type": "episode",
        "range": [1, 20],
        "n_occurrences": [0, 20],
        "static": False,
    },
    "ventilated": {
        "type": "episode",
        "range": [1, 20],
        "n_occurrences": [0, 20],
        "static": False,
    },
    "lab_ddimer": {"type": "lab", "threshold": 2, "range": [0, 4], "static": False},
    "lab_aptt": {"type": "lab", "threshold": 50, "static": False},
    "obs_tidal_volume": {
        "type": "observation",
        "threshold": 6,
        "range": [2, 10],
        "static": False,
        "occurrences_per_day": [6, 30],
        "static": False,
    },
    "obs_pmax": {
        "type": "observation",
        "threshold": 30,
        "range": [10, 100],
        "occurrences_per_day": [6, 30],
        "static": False,
    },
    "obs_fio2": {
        "type": "observation",
        "range": [0, 1.0],
        "occurrences_per_day": [6, 30],
        "static": False,
    },
    "obs_peep": {
        "type": "observation",
        "range": [4, 20],
        "occurrences_per_day": [6, 30],
        "static": False,
    },
    "obs_oxygenation_index": {
        "type": "observation",
        "range": [100, 200],
        "occurrences_per_day": [6, 30],
        "static": False,
    },
}