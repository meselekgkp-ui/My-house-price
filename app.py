import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. CUSTOM CLASSES (PFLICHT FÜR DAS MODELL) 
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
# 2. KONFIGURATION & DATEN LADEN [cite: 2, 4]
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", layout="wide")

@st.cache_data
def load_all_data():
    try:
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            gd = json.load(f)
        rev = {}
        for bl, städte in gd.items():
            for stadt, plzs in städte.items():
                for p in plzs: rev[str(p)] = (stadt, bl)
        return gd, rev
    except: return {}, {}

GEO_DATA, PLZ_LOOKUP = load_all_data()

# Session State Initialisierung
if 's_bl' not in st.session_state: st.session_state.s_bl = "Bayern"
if 's_stadt' not in st.session_state: st.session_state.s_stadt = "München"
if 's_plz' not in st.session_state: st.session_state.s_plz = "80331"

def sync():
    """Wird aufgerufen, wenn die PLZ geändert wird."""
    p = st.session_state.in_plz
    if p in PLZ_LOOKUP:
        stadt, bl = PLZ_LOOKUP[p]
        st.session_state.s_bl = bl
        st.session_state.s_stadt = stadt
        st.session_state.s_plz = p

# ==============================================================================
# 3. DESIGN & CSS
# ==============================================================================
st.markdown("""
    <style>
    .stApp { background-color: #ffffff !important; }
    h1, h2, h3, p, label { color: #262730 !important; }
    /* Button Style */
    div.stButton > button:first-child {
        background-color: #007BFF !important;
        color: white !important;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
    }
    /* Wohnungstyp schräg stellen */
    div[data-testid="stSelectbox"]:nth-of-type(5) select {
        transform: skewX(-10deg);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. STANDORT (AUSSERHALB DES FORMULARS FÜR SOFORT-SYNC)
# ==============================================================================
st.title("Intelligente Immobiliensuche 🏠")

st.markdown("### 1. Standort")
plz_in = st.text_input("PLZ eingeben (tippen zum Synchronisieren)", 
                       value=st.session_state.s_plz, 
                       key="in_plz", 
                       on_change=sync)

col_bl, col_st = st.columns(2)
bl_list = sorted(list(GEO_DATA.keys()))
with col_bl:
    sel_bl = st.selectbox("Bundesland", bl_list, 
                          index=bl_list.index(st.session_state.s_bl) if st.session_state.s_bl in bl_list else 0)

stadt_list = sorted(list(GEO_DATA.get(sel_bl, {}).keys()))
with col_st:
    sel_stadt = st.selectbox("Stadt", stadt_list, 
                             index=stadt_list.index(st.session_state.s_stadt) if st.session_state.s_stadt in stadt_list else 0)

# Synchronisation bei manueller Auswahl von Stadt/Land
st.session_state.s_bl = sel_bl
st.session_state.s_stadt = sel_stadt

st.markdown("---")

# ==============================================================================
# 5. WEITERE DATEN (IN EINEM FORMULAR)
# ==============================================================================
with st.form("objekt_details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 2. Daten")
        space = st.number_input("Wohnfläche (m²)", 10, 500, 70)
        zimmer = st.number_input("Zimmer", 1.0, 10.0, 3.0, step=0.5)
        etage = st.number_input("Etage (0=EG)", -1, 25, 1)
        baujahr = st.number_input("Baujahr", 1900, 2025, 2000)

    with col2:
        st.markdown("### 3. Ausstattung")
        HEATING = {"Zentralheizung": "central_heating", "Fernwärme": "district_heating", "Gas": "gas_heating", "Fußboden": "floor_heating"}
        COND = {"Gepflegt": "well_kept", "Neuwertig": "mint_condition", "Saniert": "refurbished", "Modernisiert": "modernized"}
        QUAL = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury"}
        TYPE = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Erdgeschoss": "ground_floor", "Maisonette": "maisonette"}
        
        heiz = st.selectbox("Heizung", list(HEATING.keys()))
        zust = st.selectbox("Zustand", list(COND.keys()))
        qual = st.selectbox("Qualität", list(QUAL.keys()))
        wtyp = st.selectbox("Wohnungstyp", list(TYPE.keys()))

    st.markdown("### 4. Extras")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1: balk = st.checkbox("Balkon")
    with e2: kuec = st.checkbox("Einbauküche")
    with e3: aufz = st.checkbox("Aufzug")
    with e4: gart = st.checkbox("Garten")
    with e5: kell = st.checkbox("Keller")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("PREIS VORHERSAGEN 🚀")

# ==============================================================================
# 6. MODELL-VORHERSAGE
# ==============================================================================
if submit_btn:
    try:
        # Modell laden - achte darauf, dass der Name exakt wie in GitHub ist! 
        model = joblib.load('mzyana_lightgbm_model.pkl')
        
        # Daten vorbereiten
        df = pd.DataFrame({
            'date': [pd.to_datetime(datetime.now())],
            'livingSpace': [float(space)],
            'noRooms': [float(zimmer)],
            'floor': [float(etage)],
            'regio1': [sel_bl],
            'regio2': [sel_stadt],
            'heatingType': [HEATING[heiz]],
            'condition': [COND[zust]],
            'interiorQual': [QUAL[qual]],
            'typeOfFlat': [TYPE[wtyp]],
            'geo_plz': [str(plz_in)],
            'balcony': [balk],
            'lift': [aufz],
            'hasKitchen': [kuec],
            'garden': [gart],
            'cellar': [kell],
            'yearConstructed': [float(baujahr)],
            'condition_was_missing': [0],
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0],
            'yearConstructed_was_missing': [0]
        })

        pred = model.predict(df)[0]
        st.success(f"Voraussichtliche Kaltmiete: {pred:,.2f} €")
        
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")
