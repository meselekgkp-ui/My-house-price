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
# 2. DESIGN & CSS (Style)
# ==============================================================================
st.set_page_config(page_title="Mietpreis-KI", layout="wide", page_icon="🏢")

st.markdown("""
    <style>
    /* Grunddesign */
    .stApp { background-color: #ffffff; }
    h1, h2, h3, p, label { color: #333333 !important; }
    
    /* WUNSCH: Schräge Eingabefelder (CSS Trick für Streamlit Inputs) */
    .stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input {
        border: 1px solid #0068C9 !important;
        border-radius: 5px;
    }

    /* Button Design */
    .stButton>button {
        background-color: #0068C9;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 1.2rem;
    }
    .stButton>button:hover {
        background-color: #004B91;
    }
    
    /* Ergebnis Box */
    .result-box {
        padding: 20px;
        background-color: #f0f8ff;
        border-left: 5px solid #0068C9;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATENBANK & LOGIK
# ==============================================================================

@st.cache_data
def load_geo_data():
    try:
        # Versucht die Datei geo_data.json zu laden
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

GEO_DATA = load_geo_data()

# ==============================================================================
# 4. APP INTERFACE (HIER PASSIERT DIE INTERAKTION)
# ==============================================================================

st.title("Mietpreis-Vorhersage KI 🤖")
st.write("Wähle die Daten aus – die Felder aktualisieren sich automatisch.")

if GEO_DATA is None:
    st.error("❌ FEHLER: 'geo_data.json' nicht gefunden. Bitte lade die Datei auf GitHub hoch.")
    st.stop()

with st.form("search_form"):
    
    # ---------------------------------------------------------
    # HIER IST DEINE LOGIK (INTERAKTION DER FELDER)
    # Streamlit macht das automatisch: Wenn "state" sich ändert,
    # lädt das Skript neu und füllt "available_cities" neu.
    # ---------------------------------------------------------
    
    st.subheader("1. Standort")
    
    # SCHRITT 1: Bundesland wählen
    all_states = sorted(list(GEO_DATA.keys()))
    state = st.selectbox("Bundesland", all_states)

    # SCHRITT 2: Städte laden, die NUR zu diesem Bundesland gehören
    # GEO_DATA[state] holt nur die Städte des gewählten Bundeslandes
    available_cities = sorted(list(GEO_DATA[state].keys()))
    city = st.selectbox("Stadt / Landkreis", available_cities)

    # SCHRITT 3: PLZ laden, die NUR zu dieser Stadt gehören
    # GEO_DATA[state][city] holt nur die PLZ dieser Stadt
    available_plzs = sorted(GEO_DATA[state][city])
    plz = st.selectbox("Postleitzahl", available_plzs)

    st.info(f"📍 Auswahl: {city} in {state} (PLZ: {plz})")
    
    st.markdown("---")
    
    # Weitere Eingabefelder
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Die Wohnung")
        living_space = st.number_input("Wohnfläche (m²)", min_value=10, max_value=500, value=60, step=1)
        rooms = st.number_input("Zimmer", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        floor = st.number_input("Etage (0 = EG)", min_value=-1, max_value=20, value=1, step=1)
        year = st.number_input("Baujahr", min_value=1900, max_value=2025, value=2000, step=1)
        
    with col2:
        st.subheader("3. Details")
        # Mappings für das Modell
        HEATING_MAP = {"Zentralheizung": "central_heating", "Fernwärme": "district_heating", "Gas": "gas_heating", "Fußboden": "floor_heating"}
        CONDITION_MAP = {"Gepflegt": "well_kept", "Neuwertig": "mint_condition", "Erstbezug": "first_time_use", "Modernisiert": "modernized"}
        QUAL_MAP = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury", "Einfach": "simple"}
        TYPE_MAP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Erdgeschoss": "ground_floor", "Maisonette": "maisonette"}
        
        heating = st.selectbox("Heizung", list(HEATING_MAP.keys()))
        condition = st.selectbox("Zustand", list(CONDITION_MAP.keys()))
        quality = st.selectbox("Ausstattung", list(QUAL_MAP.keys()))
        flat_type = st.selectbox("Wohnungstyp", list(TYPE_MAP.keys()))

    st.subheader("4. Extras")
    c1, c2, c3 = st.columns(3)
    with c1: has_balcony = st.checkbox("Balkon")
    with c2: has_kitchen = st.checkbox("Einbauküche")
    with c3: has_lift = st.checkbox("Aufzug")
    has_garden = False # Default Werte falls nicht gefragt
    has_cellar = True

    st.markdown("<br>", unsafe_allow_html=True)
    
    # DER BUTTON
    submit = st.form_submit_button("Miete berechnen 🚀")

# ==============================================================================
# 5. BERECHNUNG NACH DEM KLICK
# ==============================================================================
if submit:
    try:
        # Modell laden
        model = joblib.load('mzyana_lightgbm_model.pkl') # Dateiname muss EXAKT stimmen!
        
        # Eingabe für das Modell vorbereiten
        input_data = pd.DataFrame({
            'date': [pd.to_datetime(datetime.now())],
            'livingSpace': [float(living_space)],
            'noRooms': [float(rooms)],
            'floor': [float(floor)],
            'regio1': [state],
            'regio2': [city],
            'heatingType': [HEATING_MAP[heating]],
            'condition': [CONDITION_MAP[condition]],
            'interiorQual': [QUAL_MAP[quality]],
            'typeOfFlat': [TYPE_MAP[flat_type]],
            'geo_plz': [str(plz)], # PLZ muss String sein
            'balcony': [has_balcony],
            'lift': [has_lift],
            'hasKitchen': [has_kitchen],
            'garden': [has_garden],
            'cellar': [has_cellar],
            'yearConstructed': [float(year)],
            # Dummys für fehlende Spalten (falls das Modell sie braucht)
            'condition_was_missing': [0], 'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0], 'yearConstructed_was_missing': [0]
        })

        # Vorhersage
        prediction = model.predict(input_data)[0]

        # Anzeige
        st.markdown(f"""
        <div class="result-box">
            <h3>Geschätzte Kaltmiete:</h3>
            <h1 style="color: #0068C9; font-size: 3em;">{prediction:,.2f} €</h1>
            <p>für {city}, {living_space} m²</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")
        st.info("Tipp: Prüfe, ob die Datei 'mzyana_lightgbm_model.pkl' richtig hochgeladen ist.")
