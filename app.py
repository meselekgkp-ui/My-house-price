import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# 1. NOTWENDIGE KLASSEN FÜR DAS MODELL (Kopiert aus deinem Notebook)
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

# 2. SEITEN-KONFIGURATION & DESIGN
st.set_page_config(page_title="Mzyana Rent AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    /* DESIGN: Button mit hellem Text */
    div.stButton > button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    /* DESIGN: Wohnungstyp schräg gestellt */
    div[data-testid="stSelectbox"]:nth-of-type(4) {
        transform: skewX(-10deg);
        border: 1px solid #007BFF;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True) # Hier lag der Fehler in deinen Logs!

# 3. DATEN LADEN & SYNC-LOGIK
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

# Session State für Synchronisation
if 'bl' not in st.session_state: st.session_state.bl = "Bayern"
if 'st' not in st.session_state: st.session_state.st = "München"
if 'plz' not in st.session_state: st.session_state.plz = "80331"

def sync_plz():
    p = st.session_state.plz_in
    if p in PLZ_MAP:
        stadt, land = PLZ_MAP[p]
        st.session_state.bl, st.session_state.st, st.session_state.plz = land, stadt, p

# 4. DAS INTERFACE
st.title("Intelligente Immobiliensuche")

# Standort (Synchronisiert)
st.subheader("1. Standort")
plz = st.text_input("PLZ eingeben (Tippe z.B. 80331)", value=st.session_state.plz, key="plz_in", on_change=sync_plz)

col_bl, col_st = st.columns(2)
bl_liste = sorted(list(GEO_DATA.keys()))
with col_bl:
    sel_bl = st.selectbox("Bundesland", bl_liste, index=bl_liste.index(st.session_state.bl) if st.session_state.bl in bl_liste else 0)
    st.session_state.bl = sel_bl

staedte = sorted(list(GEO_DATA.get(sel_bl, {}).keys()))
with col_st:
    sel_st = st.selectbox("Stadt", staedte, index=staedte.index(st.session_state.st) if st.session_state.st in staedte else 0)
    st.session_state.st = sel_st

st.markdown("---")

# Details (In einem Formular)
with st.form("immo_details"):
    st.subheader("2. Wohnungsdetails")
    c1, c2 = st.columns(2)
    with c1:
        flaeche = st.number_input("Wohnfläche (m²)", 10, 500, 60)
        zimmer = st.number_input("Zimmer (Anzahl)", 1.0, 10.0, 2.0, step=0.5)
    with c2:
        etage = st.number_input("Etage (0=EG)", -1, 30, 1)
        baujahr = st.number_input("Baujahr", 1900, 2026, 2000)

    st.subheader("3. Ausstattung")
    # Mappings für dein Modell
    HEIZ = {"Zentral": "central_heating", "Gas": "gas_heating", "Fernwärme": "district_heating"}
    ZUST = {"Gepflegt": "well_kept", "Modernisiert": "modernized", "Erstbezug": "first_time_use"}
    TYP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Maisonette": "maisonette"}
    
    col_h, col_z = st.columns(2)
    with col_h: h_wahl = st.selectbox("Heizung", list(HEIZ.keys()))
    with col_z: t_wahl = st.selectbox("Wohnungstyp", list(TYP.keys()))

    st.subheader("4. Extras")
    e1, e2, e3 = st.columns(3)
    with e1: balk = st.checkbox("Balkon")
    with e2: kuech = st.checkbox("Einbauküche")
    with e3: aufz = st.checkbox("Aufzug")

    submit = st.form_submit_button("JETZT PREIS VORHERSAGEN 🚀")

# 5. VORHERSAGE
if submit:
    try:
        # Lade dein Modell (Achte darauf, dass final_model.pkl auf GitHub liegt!)
        model = joblib.load('final_model.pkl')
        
        # Daten vorbereiten
        df = pd.DataFrame({
            'date': [pd.to_datetime(datetime.now())], 'livingSpace': [float(flaeche)],
            'noRooms': [float(zimmer)], 'floor': [float(etage)], 'regio1': [sel_bl],
            'regio2': [sel_st], 'heatingType': [HEIZ[h_wahl]], 'condition': [ZUST["Gepflegt"]],
            'interiorQual': ["normal"], 'typeOfFlat': [TYP[t_wahl]], 'geo_plz': [str(plz)],
            'balcony': [balk], 'lift': [aufz], 'hasKitchen': [kuech], 'garden': [False],
            'cellar': [True], 'yearConstructed': [float(baujahr)],
            'condition_was_missing': [0], 'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0], 'yearConstructed_was_missing': [0]
        })

        preis = model.predict(df)[0]
        st.success(f"Voraussichtliche Kaltmiete: **{preis:,.2f} €**")
        st.balloons()
    except Exception as e:
        st.error(f"Fehler: {e}")
