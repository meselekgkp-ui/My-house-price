import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. MODELL-KLASSEN (Diese müssen exakt so bleiben, damit das Modell lädt)
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
# 2. DATEN & SYNC-LOGIK
# ==============================================================================
@st.cache_data
def load_geo():
    if not os.path.exists('geo_data.json'): return {}, {}
    with open('geo_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rev = {}
    for bl, staedte in data.items():
        for stadt, plzs in staedte.items():
            for p in plzs: rev[str(p)] = (stadt, bl)
    return data, rev

GEO_DATA, PLZ_MAP = load_geo()

# Initialisierung der Auswahl
if 'bl' not in st.session_state: st.session_state.bl = "Bayern"
if 'st' not in st.session_state: st.session_state.st = "München"
if 'plz' not in st.session_state: st.session_state.plz = "80331"

def on_plz_change():
    p = st.session_state.plz_input
    if p in PLZ_MAP:
        stadt, land = PLZ_MAP[p]
        st.session_state.bl, st.session_state.st, st.session_state.plz = land, stadt, p

# ==============================================================================
# 3. DESIGN (CSS)
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", layout="centered")

st.markdown("""
    <style>
    /* Button: Blauer Hintergrund, weißer Text */
    div.stButton > button {
        background-color: #007BFF !important;
        color: white !important;
        font-weight: bold;
        height: 50px;
        width: 100%;
        border-radius: 10px;
        border: none;
    }
    /* Wohnungstyp: Schräg gestellt */
    div[data-testid="stSelectbox"]:nth-of-type(4) {
        transform: skewX(-10deg);
        border: 1px solid #007BFF;
        border-radius: 5px;
    }
    /* Allgemeine Verschönerung */
    .stApp { background-color: #F8F9FA; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. DAS FORMULAR
# ==============================================================================
st.title("Mzyana AI: Mietpreis-Vorhersage 🏠")

# A. STANDORT (Synchron)
st.subheader("1. Wo liegt die Wohnung?")
plz_in = st.text_input("PLZ eingeben", value=st.session_state.plz, key="plz_input", on_change=on_plz_change)

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

# B. DETAILS (In einem Formular für Ordnung)
with st.form("details"):
    st.subheader("2. Details zur Wohnung")
    c1, c2 = st.columns(2)
    with c1:
        flaeche = st.number_input("Wohnfläche (m²)", min_value=10, max_value=500, value=60)
        zimmer = st.number_input("Zimmer", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    with c2:
        etage = st.number_input("Etage (0=EG)", min_value=-1, max_value=30, value=1)
        baujahr = st.number_input("Baujahr", min_value=1900, max_value=2026, value=2000)

    st.subheader("3. Ausstattung")
    # Mappings für das Modell
    HEIZ = {"Zentral": "central_heating", "Gas": "gas_heating", "Fußboden": "floor_heating", "Fernwärme": "district_heating"}
    ZUST = {"Gepflegt": "well_kept", "Modernisiert": "modernized", "Erstbezug": "first_time_use"}
    QUAL = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury"}
    TYP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Maisonette": "maisonette"}

    col_h, col_z = st.columns(2)
    with col_h:
        h_wahl = st.selectbox("Heizung", list(HEIZ.keys()))
        q_wahl = st.selectbox("Qualität", list(QUAL.keys()))
    with col_z:
        z_wahl = st.selectbox("Zustand", list(ZUST.keys()))
        t_wahl = st.selectbox("Wohnungstyp", list(TYP.keys())) # Dieser wird schräg angezeigt

    st.subheader("4. Extras")
    ex1, ex2, ex3 = st.columns(3)
    with ex1: balk = st.checkbox("Balkon")
    with ex2: kuech = st.checkbox("Küche")
    with ex3: aufz = st.checkbox("Aufzug")

    submit = st.form_submit_button("JETZT MIETPREIS BERECHNEN 🚀")

# ==============================================================================
# 5. BERECHNUNG
# ==============================================================================
if submit:
    try:
        # Hier laden wir dein neues 'final_model.pkl'
        model = joblib.load('final_model.pkl')
        
        # Eingabedaten für das Modell bauen
        input_data = pd.DataFrame({
            'date': [pd.to_datetime(datetime.now())],
            'livingSpace': [float(flaeche)],
            'noRooms': [float(zimmer)],
            'floor': [float(etage)],
            'regio1': [sel_bl],
            'regio2': [sel_st],
            'heatingType': [HEIZ[h_wahl]],
            'condition': [ZUST[z_wahl]],
            'interiorQual': [QUAL[q_wahl]],
            'typeOfFlat': [TYP[t_wahl]],
            'geo_plz': [str(st.session_state.plz_input)],
            'balcony': [balk],
            'lift': [aufz],
            'hasKitchen': [kuech],
            'garden': [False],
            'cellar': [True],
            'yearConstructed': [float(baujahr)],
            'condition_was_missing': [0],
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0],
            'yearConstructed_was_missing': [0]
        })

        prediction = model.predict(input_data)[0]

        st.balloons()
        st.success(f"Die geschätzte Kaltmiete beträgt: **{prediction:,.2f} €**")
        
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")
