import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. HELFER-KLASSEN (NICHT LÖSCHEN!)
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
# 2. DESIGN & SETUP
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div.stButton > button {
        background-color: #007BFF !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        border: none;
    }
    div[data-testid="stSelectbox"]:nth-of-type(4) > div > div {
        transform: skewX(-10deg);
        border: 2px solid #007BFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATEN LADEN & SYNC-LOGIK
# ==============================================================================
@st.cache_data
def load_data():
    if not os.path.exists('geo_data.json'): return {}, {}
    with open('geo_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rev = {}
    for bl, staedte in data.items():
        for stadt, plzs in staedte.items():
            for p in plzs: rev[str(p)] = (stadt, bl)
    return data, rev

GEO_DATA, PLZ_MAP = load_data()

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
# 4. APP INTERFACE
# ==============================================================================
st.title("Intelligente Immobiliensuche 🏡")

# Standort (Live-Update)
st.subheader("1. Standort")
plz = st.text_input("PLZ (z.B. 80331)", value=st.session_state.plz, key="plz_in", on_change=sync_plz)

c1, c2 = st.columns(2)
bl_keys = sorted(list(GEO_DATA.keys()))
with c1:
    idx_bl = bl_keys.index(st.session_state.bl) if st.session_state.bl in bl_keys else 0
    sel_bl = st.selectbox("Bundesland", bl_keys, index=idx_bl)
    st.session_state.bl = sel_bl

st_keys = sorted(list(GEO_DATA.get(sel_bl, {}).keys()))
with c2:
    idx_st = st_keys.index(st.session_state.st) if st.session_state.st in st_keys else 0
    sel_st = st.selectbox("Stadt", st_keys, index=idx_st)
    st.session_state.st = sel_st

st.markdown("---")

# Details Formular
with st.form("main_form"):
    st.subheader("2. Details")
    c_a, c_b = st.columns(2)
    with c_a:
        flaeche = st.number_input("Wohnfläche (m²)", 10, 500, 60)
        zimmer = st.number_input("Zimmer", 1.0, 10.0, 2.0, step=0.5)
    with c_b:
        etage = st.number_input("Etage (0=EG)", -1, 30, 1)
        baujahr = st.number_input("Baujahr", 1900, 2026, 2000)

    st.subheader("3. Ausstattung")
    HEIZ = {"Zentral": "central_heating", "Gas": "gas_heating", "Fernwärme": "district_heating"}
    ZUST = {"Gepflegt": "well_kept", "Modernisiert": "modernized", "Erstbezug": "first_time_use"}
    TYP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Maisonette": "maisonette", "Loft": "loft"}
    
    col_h, col_t = st.columns(2)
    with col_h: h_wahl = st.selectbox("Heizung", list(HEIZ.keys()))
    with col_t: t_wahl = st.selectbox("Wohnungstyp", list(TYP.keys()))

    st.subheader("4. Extras")
    chk1, chk2, chk3 = st.columns(3)
    with chk1: balk = st.checkbox("Balkon")
    with chk2: kuech = st.checkbox("Einbauküche")
    with chk3: aufz = st.checkbox("Aufzug")

    submit = st.form_submit_button("JETZT PREIS BERECHNEN 🚀")

# ==============================================================================
# 5. VORHERSAGE
# ==============================================================================
if submit:
    # Automatische Suche nach dem Modell
    files = ['final_model.pkl', 'mzyana_lightgbm_model.pkl']
    model = None
    for f in files:
        if os.path.exists(f):
            try:
                model = joblib.load(f)
                break
            except: continue
            
    if model is None:
        st.error("❌ FEHLER: Modell nicht gefunden! Bitte lade 'mzyana_lightgbm_model.pkl' oder 'final_model.pkl' hoch.")
        st.write("Dateien im Ordner:", os.listdir())
    else:
        try:
            df = pd.DataFrame({
                'date': [pd.to_datetime(datetime.now())], 'livingSpace': [float(flaeche)],
                'noRooms': [float(zimmer)], 'floor': [float(etage)], 'regio1': [sel_bl],
                'regio2': [sel_st], 'heatingType': [HEIZ.get(h_wahl, "central_heating")], 
                'condition': [ZUST.get("Gepflegt")], 
                'interiorQual': ["normal"], 'typeOfFlat': [TYP.get(t_wahl, "apartment")], 'geo_plz': [str(plz)],
                'balcony': [balk], 'lift': [aufz], 'hasKitchen': [kuech], 'garden': [False],
                'cellar': [True], 'yearConstructed': [float(baujahr)],
                'condition_was_missing': [0], 'interiorQual_was_missing': [0],
                'heatingType_was_missing': [0], 'yearConstructed_was_missing': [0]
            })

            preis = model.predict(df)[0]
            st.success(f"Geschätzte Miete: {preis:,.2f} €")
            st.balloons()
        except Exception as e:
            st.error(f"Fehler bei der Berechnung: {e}")
