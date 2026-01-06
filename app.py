import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. CUSTOM CLASSES (PFLICHT)
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
# 2. DESIGN & CSS (TOTAL-FIX FÜR SICHTBARKEIT)
# ==============================================================================

st.set_page_config(page_title="Mietpreis-Expertensystem", layout="wide", page_icon="🏢")

# Dieses CSS überschreibt ALLE Streamlit-Standardfarben
st.markdown("""
    <style>
    /* 1. Haupt-Hintergrund immer Weiß */
    .stApp {
        background-color: #ffffff !important;
    }

    /* 2. Alle Texte (Überschriften, Labels, Paragraphen) immer Dunkelgrau */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #262730 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 3. Eingabefelder (Inputs) reparieren */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        color: #000000 !important; /* Text im Feld schwarz */
        background-color: #F0F2F6 !important; /* Hintergrund leicht grau */
        border: 1px solid #D6D6D6 !important;
    }
    
    /* 4. Dropdown-Menüs Lesbarkeit */
    div[role="listbox"] ul {
        background-color: #ffffff !important;
    }
    div[role="option"] {
        color: #000000 !important;
    }
    div[role="option"]:hover {
        background-color: #E6F3FF !important;
    }

    /* 5. Button Styling */
    .stButton>button {
        background-color: #0068C9 !important;
        color: white !important;
        font-size: 18px !important;
        border-radius: 8px !important;
        height: 50px !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #004B91 !important;
    }

    /* 6. Ergebnis-Box Design */
    .result-container {
        padding: 30px;
        background-color: #F9F9F9;
        border-left: 6px solid #0068C9;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATENBANK & LOGIK
# ==============================================================================

# Datenbank laden
@st.cache_data # Cache damit es schneller lädt
def load_geo_data():
    try:
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

GEO_DATA = load_geo_data()

# Mappings
HEATING_MAP = {
    "Zentralheizung": "central_heating", "Fernwärme": "district_heating", "Gas-Heizung": "gas_heating", 
    "Etagenheizung": "self_contained_central_heating", "Fußbodenheizung": "floor_heating", 
    "Ölheizung": "oil_heating", "Wärmepumpe": "heat_pump", "Holzpelletheizung": "wood_pellet_heating",
    "Andere": "central_heating"
}
CONDITION_MAP = {
    "Gepflegt": "well_kept", "Erstbezug": "first_time_use", "Saniert": "refurbished", 
    "Vollständig renoviert": "fully_renovated", "Neuwertig": "mint_condition", 
    "Modernisiert": "modernized", "Erstbezug nach Sanierung": "first_time_use_after_refurbishment", 
    "Andere": "negotiable"
}
TYPE_MAP = {
    "Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Erdgeschoss": "ground_floor", 
    "Maisonette": "maisonette", "Hochparterre": "raised_ground_floor", "Penthouse": "penthouse", 
    "Souterrain": "half_basement", "Andere": "apartment"
}
QUAL_MAP = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury", "Einfach": "simple"}

# ==============================================================================
# 4. APP INTERFACE
# ==============================================================================

st.title("Mietpreis-Expertensystem")
st.markdown("---")

if GEO_DATA is None:
    st.error("❌ FEHLER: Die Datei 'geo_data.json' wurde nicht gefunden. Bitte laden Sie diese auf GitHub hoch.")
    st.stop()


# --- STANDORT (AUSSERHALB DES FORMS, DAMIT SYNC FUNKTIONIERT) ---
def build_plz_index(data):
    index = {}
    for state_key, cities in data.items():
        for city_key, plzs in cities.items():
            for plz_val in plzs:
                if plz_val not in index:
                    index[plz_val] = (state_key, city_key)
    return index

PLZ_INDEX = build_plz_index(GEO_DATA)

def init_location_state():
    if "s_state" not in st.session_state:
        st.session_state.s_state = sorted(GEO_DATA.keys())[0]
    if "s_city" not in st.session_state:
        st.session_state.s_city = sorted(GEO_DATA[st.session_state.s_state].keys())[0]
    if "s_plz" not in st.session_state:
        st.session_state.s_plz = sorted(GEO_DATA[st.session_state.s_state][st.session_state.s_city])[0]

def update_cities():
    state_key = st.session_state.s_state
    cities = sorted(GEO_DATA[state_key].keys())
    if st.session_state.s_city not in cities:
        st.session_state.s_city = cities[0]
    update_plz_list()

def update_plz_list():
    state_key = st.session_state.s_state
    city_key = st.session_state.s_city
    plzs = sorted(GEO_DATA[state_key][city_key])
    if st.session_state.s_plz not in plzs:
        st.session_state.s_plz = plzs[0]

def sync_from_plz():
    plz_val = st.session_state.s_plz
    if plz_val in PLZ_INDEX:
        state_key, city_key = PLZ_INDEX[plz_val]
        st.session_state.s_state = state_key
        st.session_state.s_city = city_key

init_location_state()

st.markdown("### 1. Standort")
all_states = sorted(list(GEO_DATA.keys()))
state = st.selectbox("Bundesland", all_states, key="s_state", on_change=update_cities)

available_cities = sorted(list(GEO_DATA[state].keys()))
city = st.selectbox("Stadt / Landkreis (Tippen zum Suchen)", available_cities, key="s_city", on_change=update_plz_list)

available_plzs = sorted(GEO_DATA[state][city])
plz = st.selectbox("Postleitzahl", available_plzs, key="s_plz", on_change=sync_from_plz)

st.caption(f"Gewaehlt: {plz} {city}, {state}")
st.markdown("---")

# --- HAUPTFORMULAR ---
with st.form("main_form"):
    # 2. OBJEKTDATEN & AUSSTATTUNG
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 2. Gebäudedaten")
        living_space = st.number_input("Wohnfläche (m²)", 10, 600, 75, step=1)
        rooms = st.number_input("Zimmer", 1, 15, 3, step=1)
        floor = st.number_input("Etage", 0, 40, 1, step=1)
        year = st.number_input("Baujahr", 1900, 2025, 1995, step=1)
        
    with col2:
        st.markdown("### 3. Qualität & Zustand")
        heating = st.selectbox("Heizung", list(HEATING_MAP.keys()))
        condition = st.selectbox("Zustand", list(CONDITION_MAP.keys()))
        quality = st.selectbox("Qualität", list(QUAL_MAP.keys()))
        flat_type = st.selectbox("Wohnungstyp", list(TYPE_MAP.keys()))

    # 4. EXTRAS
    st.markdown("### 4. Extras")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: has_balcony = st.checkbox("Balkon")
    with c2: has_lift = st.checkbox("Aufzug")
    with c3: has_kitchen = st.checkbox("Einbauküche")
    with c4: has_garden = st.checkbox("Garten")
    with c5: has_cellar = st.checkbox("Keller")

    # Datum (Versteckt oder Default Heute)
    date_val = datetime.now()

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("MIETPREIS BERECHNEN")

# ==============================================================================
# 5. LOGIK
# ==============================================================================

if submit:
    try:
        model = joblib.load('mzyana_lightgbm_model.pkl')
        
        # Dataframe exakt wie im Training
        df_input = pd.DataFrame({
            'date': [pd.to_datetime(date_val)],
            'livingSpace': [float(living_space)],
            'noRooms': [float(rooms)],
            'floor': [float(floor)],
            'regio1': [state],
            'regio2': [city],
            'heatingType': [HEATING_MAP[heating]],
            'condition': [CONDITION_MAP[condition]],
            'interiorQual': [QUAL_MAP[quality]],
            'typeOfFlat': [TYPE_MAP[flat_type]],
            'geo_plz': [str(plz)],
            'balcony': [has_balcony],
            'lift': [has_lift],
            'hasKitchen': [has_kitchen],
            'garden': [has_garden],
            'cellar': [has_cellar],
            'yearConstructed': [float(year)],
            'condition_was_missing': [0],
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0],
            'yearConstructed_was_missing': [0]
        })

        pred = model.predict(df_input)[0]

        # Ergebnis
        st.markdown(f"""
        <div class="result-container">
            <h3 style="color: #555; margin: 0;">Geschätzte Gesamtmiete</h3>
            <h1 style="color: #0068C9; font-size: 60px; margin: 10px 0;">{pred:,.2f} €</h1>
            <p style="color: #888;">Berechnet für {city} ({plz}) • {living_space} m²</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Fehler: {e}")

st.markdown("---")
st.caption("Masterprojekt Prof. Wahl | Data Science 2025")

