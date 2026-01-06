import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. NOTWENDIGE KLASSEN (Damit das Modell funktioniert)
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
# 2. SEITEN-DESIGN & CSS
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    /* Button: Blau mit HELLEM (weißem) Text */
    div.stButton > button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        height: 50px !important;
        width: 100%;
        border: none;
        font-size: 1.1rem;
    }
    div.stButton > button:hover {
        background-color: #0056b3 !important;
    }

    /* Wohnungstyp: SCHRÄG gestellt (Skew) */
    div[data-testid="stSelectbox"]:nth-of-type(4) > div > div {
        transform: skewX(-10deg);
        border: 2px solid #007BFF !important;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATEN & LOGIK
# ==============================================================================
@st.cache_data
def load_geo_data():
    if os.path.exists('geo_data.json'):
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        reverse = {}
        for bl, städte in data.items():
            for stadt, plzs in städte.items():
                for p in plzs: reverse[str(p)] = (stadt, bl)
        return data, reverse
    return {}, {}

GEO_DATA, PLZ_MAP = load_geo_data()

# Session State Initialisierung für Synchronisation
if 'bl' not in st.session_state: st.session_state.bl = "Bayern"
if 'st' not in st.session_state: st.session_state.st = "München"
if 'plz' not in st.session_state: st.session_state.plz = "80331"

def sync_plz():
    p = st.session_state.plz_in
    if p in PLZ_MAP:
        stadt, land = PLZ_MAP[p]
        st.session_state.bl = land
        st.session_state.st = stadt
        st.session_state.plz = p

# ==============================================================================
# 4. DAS INTERFACE
# ==============================================================================
st.title("Intelligente Immobiliensuche 🏢")

# --- STANDORT ---
st.subheader("1. Standort")
plz_val = st.text_input("Postleitzahl (PLZ)", value=st.session_state.plz, key="plz_in", on_change=sync_plz)

c_bl, c_st = st.columns(2)
bl_keys = sorted(list(GEO_DATA.keys()))
with c_bl:
    idx_bl = bl_keys.index(st.session_state.bl) if st.session_state.bl in bl_keys else 0
    sel_bl = st.selectbox("Bundesland", bl_keys, index=idx_bl)
    st.session_state.bl = sel_bl

st_keys = sorted(list(GEO_DATA.get(sel_bl, {}).keys()))
with c_st:
    idx_st = st_keys.index(st.session_state.st) if st.session_state.st in st_keys else 0
    sel_st = st.selectbox("Stadt", st_keys, index=idx_st)
    st.session_state.st = sel_st

st.markdown("---")

# --- DETAILS FORMULAR ---
with st.form("rent_form"):
    st.subheader("2. Wohnungsdetails")
    col1, col2 = st.columns(2)
    with col1:
        flaeche = st.number_input("Wohnfläche (m²)", 10, 500, 60)
        zimmer = st.number_input("Zimmer Anzahl", 1.0, 10.0, 2.0, step=0.5)
    with col2:
        etage = st.number_input("Etage (0=EG)", -1, 30, 1)
        baujahr = st.number_input("Baujahr", 1900, 2026, 2000)

    st.subheader("3. Ausstattung")
    HEIZ = {"Zentral": "central_heating", "Gas": "gas_heating", "Fernwärme": "district_heating"}
    ZUST = {"Gepflegt": "well_kept", "Modernisiert": "modernized", "Erstbezug": "first_time_use"}
    TYP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Maisonette": "maisonette"}
    
    col_h, col_t = st.columns(2)
    with col_h: h_wahl = st.selectbox("Heizung", list(HEIZ.keys()))
    with col_t: t_wahl = st.selectbox("Wohnungstyp", list(TYP.keys()))

    st.subheader("4. Extras")
    e1, e2, e3 = st.columns(3)
    with e1: balk = st.checkbox("Balkon")
    with e2: kuech = st.checkbox("Einbauküche")
    with e3: aufz = st.checkbox("Aufzug")

    submit = st.form_submit_button("JETZT PREIS VORHERSAGEN 🚀")

# ==============================================================================
# 5. BERECHNUNG
# ==============================================================================
if submit:
    model_path = 'mzyana_lightgbm_model.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Datei '{model_path}' wurde im Ordner nicht gefunden!")
    else:
        try:
            # Modell laden
            model = joblib.load(model_path)
            
            # DataFrame für Vorhersage
            df = pd.DataFrame({
                'date': [pd.to_datetime(datetime.now())], 'livingSpace': [float(flaeche)],
                'noRooms': [float(zimmer)], 'floor': [float(etage)], 'regio1': [sel_bl],
                'regio2': [sel_st], 'heatingType': [HEIZ[h_wahl]], 'condition': [ZUST["Gepflegt"]],
                'interiorQual': ["normal"], 'typeOfFlat': [TYP[t_wahl]], 'geo_plz': [str(plz_val)],
                'balcony': [balk], 'lift': [aufz], 'hasKitchen': [kuech], 'garden': [False],
                'cellar': [True], 'yearConstructed': [float(baujahr)],
                'condition_was_missing': [0], 'interiorQual_was_missing': [0],
                'heatingType_was_missing': [0], 'yearConstructed_was_missing': [0]
            })

            prediction = model.predict(df)[0]
            st.success(f"Voraussichtliche Kaltmiete: **{prediction:,.2f} €**")
            st.balloons()

        except Exception as e:
            st.error(f"Berechnungsfehler: {e}")
            st.info("Hinweis: Überprüfe, ob xgboost und lightgbm in der requirements.txt stehen.")
