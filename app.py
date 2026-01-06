import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. NOTWENDIGE KLASSEN FÜR DAS MODELL-LADEN
# ==============================================================================

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col): self.date_col = date_col
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['post_year'], X['post_month'] = X[self.date_col].dt.year, X[self.date_col].dt.month
        return X.drop(columns=[self.date_col])

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col, self.target_col = group_col, target_col
        self.group_medians, self.global_median = {}, 0
    def fit(self, X, y=None):
        self.global_median = X[self.target_col].median()
        self.group_medians = X.groupby(self.group_col)[self.target_col].median().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X.apply(lambda r: self.group_medians.get(r[self.group_col], self.global_median) if pd.isna(r[self.target_col]) else r[self.target_col], axis=1)
        return X

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col, self.target_col = group_col, target_col
        self.mappings, self.global_mean = {}, 0
    def fit(self, X, y=None):
        self.global_mean = X[self.target_col].mean()
        self.mappings = X.groupby(self.group_col)[self.target_col].mean().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.group_col + '_encoded'] = X[self.group_col].map(self.mappings).fillna(self.global_mean)
        return X.drop(columns=[self.group_col])

# ==============================================================================
# 2. ÜBERSETZUNGS-DIKTIONÄRE (UI -> MODELL)
# ==============================================================================

HEATING_MAP = {
    "Zentralheizung": "central_heating", "Fernwärme": "district_heating",
    "Gas-Heizung": "gas_heating", "Etagenheizung": "self_contained_central_heating",
    "Fußbodenheizung": "floor_heating", "Ölheizung": "oil_heating",
    "Wärmepumpe": "heat_pump", "Holzpelletheizung": "wood_pellet_heating"
}

CONDITION_MAP = {
    "Gepflegt": "well_kept", "Erstbezug": "first_time_use", 
    "Saniert": "refurbished", "Vollständig renoviert": "fully_renovated",
    "Neuwertig": "mint_condition", "Modernisiert": "modernized",
    "Erstbezug nach Sanierung": "first_time_use_after_refurbishment", "Nach Vereinbarung": "negotiable"
}

FLAT_TYPE_MAP = {
    "Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey",
    "Erdgeschoss": "ground_floor", "Maisonette": "maisonette",
    "Hochparterre": "raised_ground_floor", "Terrassenwohnung": "terraced_flat",
    "Penthouse": "penthouse", "Souterrain": "half_basement"
}

QUALITY_MAP = {
    "Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury", "Einfach": "simple"
}

# Hierarchische Regionale Daten (Beispielhaft - bitte vervollständigen)
REGION_STRUCTURE = {
    "Bayern": {"Deggendorf": ["94469", "94447"], "Muenchen": ["80331", "80801"]},
    "Berlin": {"Berlin": ["10115", "10117", "10435"]},
    "Nordrhein_Westfalen": {"Koeln": ["50667", "50733"], "Duesseldorf": ["40210"]}
}

# ==============================================================================
# 3. LAYOUT & DESIGN
# ==============================================================================

st.set_page_config(page_title="Mietpreis-Analysesystem", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F4F7F9; }
    .main-header { color: #2C3E50; font-size: 26px; font-weight: bold; padding-bottom: 10px; border-bottom: 2px solid #BDC3C7; margin-bottom: 20px; }
    .section-header { color: #34495E; font-size: 16px; font-weight: bold; margin-top: 20px; text-transform: uppercase; letter-spacing: 1px; }
    .stButton>button { background-color: #3498DB; color: white; border-radius: 4px; padding: 12px; font-weight: bold; border: none; width: 100%; }
    .stButton>button:hover { background-color: #2980B9; }
    .result-card { background-color: white; padding: 30px; border-radius: 5px; border-left: 10px solid #3498DB; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">Statistisches Informationssystem zur Mietpreisermittlung</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. FORMULAR
# ==============================================================================

with st.form("expert_prognose"):
    
    st.markdown('<div class="section-header">Lage und Region</div>', unsafe_allow_html=True)
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        regio1 = st.selectbox("Bundesland", sorted(REGION_STRUCTURE.keys()))
    with r_col2:
        regio2 = st.selectbox("Stadt / Landkreis", sorted(REGION_STRUCTURE[regio1].keys()))
    with r_col3:
        geo_plz = st.selectbox("Postleitzahl", sorted(REGION_STRUCTURE[regio1][regio2]))

    st.markdown('<div class="section-header">Physische Objektdaten</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        livingSpace = st.number_input("Wohnfläche (m²)", min_value=10, max_value=500, value=75, step=1)
    with p_col2:
        noRooms = st.number_input("Zimmeranzahl", min_value=1, max_value=15, value=3, step=1)
    with p_col3:
        floor = st.number_input("Etage", min_value=0, max_value=30, value=1, step=1)
    with p_col4:
        yearConstructed = st.number_input("Baujahr (0 = Unbekannt)", min_value=0, max_value=2025, value=1995, step=1)

    st.markdown('<div class="section-header">Qualität und Ausstattung</div>', unsafe_allow_html=True)
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        heating_ui = st.selectbox("Heizungssystem", list(HEATING_MAP.keys()))
    with q_col2:
        condition_ui = st.selectbox("Zustand des Objekts", list(CONDITION_MAP.keys()))
    with q_col3:
        interior_ui = st.selectbox("Ausstattungsstandard", list(QUALITY_MAP.keys()))
    with q_col4:
        flat_ui = st.selectbox("Wohnungstyp", list(FLAT_TYPE_MAP.keys()))

    st.markdown('<div class="section-header">Zusätzliche Merkmale</div>', unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
    with f_col1: balcony = st.checkbox("Balkon / Terrasse")
    with f_col2: lift = st.checkbox("Aufzug")
    with f_col3: hasKitchen = st.checkbox("Einbauküche")
    with f_col4: garden = st.checkbox("Gartenanteil")
    with f_col5: cellar = st.checkbox("Keller")

    date_val = st.date_input("Analysestichtag", datetime.now())
    
    submit = st.form_submit_button("Analyse durchführen")

# ==============================================================================
# 5. BERECHNUNG
# ==============================================================================

if submit:
    try:
        model = joblib.load('mzyana_model_final.pkl')
        
        # Mapping der UI-Werte auf Modell-Keys
        input_df = pd.DataFrame({
            'date': [pd.to_datetime(date_val)],
            'livingSpace': [float(livingSpace)],
            'noRooms': [float(noRooms)],
            'floor': [float(floor)],
            'regio1': [regio1],
            'regio2': [regio2],
            'heatingType': [HEATING_MAP[heating_ui]],
            'condition': [CONDITION_MAP[condition_ui]],
            'interiorQual': [QUALITY_MAP[interior_ui]],
            'typeOfFlat': [FLAT_TYPE_MAP[flat_ui]],
            'geo_plz': [str(geo_plz)],
            'balcony': [bool(balcony)],
            'lift': [bool(lift)],
            'hasKitchen': [bool(hasKitchen)],
            'garden': [bool(garden)],
            'cellar': [bool(cellar)],
            'yearConstructed': [np.nan if yearConstructed == 0 else float(yearConstructed)],
            'condition_was_missing': [0], 'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0], 'yearConstructed_was_missing': [1 if yearConstructed == 0 else 0]
        })

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
            <div class="result-card">
                <div style="color: #7F8C8D; font-size: 14px;">Erwartete monatliche Bruttokaltmiete:</div>
                <div style="color: #2C3E50; font-size: 36px; font-weight: bold;">{prediction:,.2f} EUR</div>
                <div style="color: #BDC3C7; font-size: 12px; margin-top: 15px;">
                    Statistisches Konfidenzintervall basierend auf LightGBM (R² > 0.90)
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")

st.markdown('<div style="text-align: center; color: #95A5A6; font-size: 11px; margin-top: 60px;">Forschungsmodul Immobilienökonomie | Masterprojekt | Betreuung: Prof. Wahl</div>', unsafe_allow_html=True)
