# --- DEBUGGING START ---
import os
st.write("📂 Aktuelles Verzeichnis:", os.getcwd())
st.write("📄 Alle Dateien hier:", os.listdir())
# --- DEBUGGING ENDE ---import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. HILFSKLASSEN (MÜSSEN VOR DEM MODELL-LADEN DEFINIERT SEIN)
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
# 2. KONFIGURATION & DATEN LADEN
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", layout="centered")

@st.cache_data
def get_data():
    try:
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Rückwärts-Suche für PLZ erstellen
        reverse_map = {}
        for bl, städte in data.items():
            for stadt, plzs in städte.items():
                for p in plzs:
                    reverse_map[str(p)] = (stadt, bl)
        return data, reverse_map
    except Exception:
        return {}, {}

GEO_DATA, PLZ_LOOKUP = get_data()

# ==============================================================================
# 3. SESSION STATE (Die Synchronisations-Logik)
# ==============================================================================
if 'sel_state' not in st.session_state: st.session_state.sel_state = "Bayern"
if 'sel_city' not in st.session_state: st.session_state.sel_city = "München"
if 'sel_plz' not in st.session_state: st.session_state.sel_plz = "80331"

def sync_from_plz():
    """Wenn PLZ eingetippt wird: Stadt & Bundesland automatisch setzen"""
    p = st.session_state.input_plz
    if p in PLZ_LOOKUP:
        stadt, bl = PLZ_LOOKUP[p]
        st.session_state.sel_state = bl
        st.session_state.sel_city = stadt
        st.session_state.sel_plz = p

# ==============================================================================
# 4. DESIGN (CSS)
# ==============================================================================
st.markdown("""
    <style>
    /* Heller Button */
    .stButton>button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 100%;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover { background-color: #0056b3 !important; }

    /* Schräger Wohnungstyp (Wir zielen auf den 4. Selectbox-Container) */
    /* Hinweis: Das ist ein Trick. Wenn sich die Reihenfolge ändert, muss das angepasst werden. */
    div[data-testid="stSelectbox"]:nth-of-type(4) > div > div {
        transform: skewX(-10deg);
        border-color: #007BFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. APP INTERFACE
# ==============================================================================
st.title("Intelligente Immobiliensuche 🏠")

with st.form("main_form"):
    
    # --- TEIL A: STANDORT (SYNCHRONISIERT) ---
    st.markdown("### 1. Standort")
    
    # PLZ Eingabe (Trigger für Auto-Fill)
    plz_input = st.text_input(
        "Postleitzahl (PLZ)", 
        value=st.session_state.sel_plz,
        key="input_plz",
        placeholder="z.B. 80331 eintippen...",
    )
    # Kleiner Hack: Wir rufen die Sync-Funktion manuell auf, wenn sich PLZ ändert,
    # aber innerhalb eines Forms greift on_change erst beim Submit. 
    # Daher prüfen wir hier direkt:
    if plz_input != st.session_state.sel_plz:
         if plz_input in PLZ_LOOKUP:
            stadt, bl = PLZ_LOOKUP[plz_input]
            st.session_state.sel_state = bl
            st.session_state.sel_city = stadt
            st.session_state.sel_plz = plz_input
            st.rerun() # Seite neu laden mit neuen Daten

    # Bundesland
    all_states = sorted(list(GEO_DATA.keys()))
    try: state_idx = all_states.index(st.session_state.sel_state)
    except: state_idx = 0
    selected_state = st.selectbox("Bundesland", all_states, index=state_idx)

    # Stadt (Abhängig vom Bundesland)
    cities_in_state = sorted(list(GEO_DATA.get(selected_state, {}).keys()))
    try: city_idx = cities_in_state.index(st.session_state.sel_city)
    except: city_idx = 0
    selected_city = st.selectbox("Stadt", cities_in_state, index=city_idx)
    
    # Update Session State für den nächsten Durchlauf
    st.session_state.sel_state = selected_state
    st.session_state.sel_city = selected_city
    
    st.markdown("---")

    # --- TEIL B: OBJEKTDATEN ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 2. Daten")
        living_space = st.number_input("Wohnfläche (m²)", 10, 500, 60)
        rooms = st.number_input("Zimmer", 1.0, 10.0, 2.0, step=0.5)
        floor = st.number_input("Etage (0=EG)", -1, 40, 1)
        year = st.number_input("Baujahr", 1900, 2025, 2000)

    with col2:
        st.markdown("### 3. Ausstattung")
        # Mappings (Deutsch -> Modell-Englisch)
        HEATING_MAP = {"Zentralheizung": "central_heating", "Fernwärme": "district_heating", "Gas": "gas_heating", "Fußboden": "floor_heating", "Etagenheizung": "self_contained_central_heating", "Wärmepumpe": "heat_pump", "Öl": "oil_heating"}
        CONDITION_MAP = {"Gepflegt": "well_kept", "Neuwertig": "mint_condition", "Erstbezug": "first_time_use", "Modernisiert": "modernized", "Saniert": "refurbished", "Renoviert": "fully_renovated"}
        QUAL_MAP = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury", "Einfach": "simple"}
        TYPE_MAP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Erdgeschoss": "ground_floor", "Maisonette": "maisonette", "Penthouse": "penthouse", "Loft": "loft"}
        
        heating = st.selectbox("Heizung", list(HEATING_MAP.keys()))
        condition = st.selectbox("Zustand", list(CONDITION_MAP.keys()))
        quality = st.selectbox("Qualität", list(QUAL_MAP.keys()))
        # Das hier ist der "schräge" Input (4. Selectbox)
        flat_type = st.selectbox("Wohnungstyp", list(TYPE_MAP.keys()))

    # --- TEIL C: EXTRAS ---
    st.markdown("### 4. Extras")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: has_balcony = st.checkbox("Balkon")
    with c2: has_kitchen = st.checkbox("Einbauküche")
    with c3: has_lift = st.checkbox("Aufzug")
    with c4: has_garden = st.checkbox("Garten")
    with c5: has_cellar = st.checkbox("Keller")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("PREIS VORHERSAGEN 🚀")

# ==============================================================================
# 6. VORHERSAGE LOGIK
# ==============================================================================
if submit:
    try:
        # 1. Modell laden
        # HINWEIS: Dateiname muss exakt stimmen. Prüfe, ob es 'mzyana_lightgbm_model.pkl' heißt.
        model = joblib.load('mzyana_lightgbm_model.pkl') 
        
        # 2. DataFrame bauen
        input_data = pd.DataFrame({
            'date': [pd.to_datetime(datetime.now())],
            'livingSpace': [float(living_space)],
            'noRooms': [float(rooms)],
            'floor': [float(floor)],
            'regio1': [selected_state],
            'regio2': [selected_city],
            'heatingType': [HEATING_MAP[heating]],
            'condition': [CONDITION_MAP[condition]],
            'interiorQual': [QUAL_MAP[quality]],
            'typeOfFlat': [TYPE_MAP[flat_type]],
            'geo_plz': [str(plz_input)],
            'balcony': [has_balcony],
            'lift': [has_lift],
            'hasKitchen': [has_kitchen],
            'garden': [has_garden],
            'cellar': [has_cellar],
            'yearConstructed': [float(year)],
            # Dummys falls nötig
            'condition_was_missing': [0], 
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0], 
            'yearConstructed_was_missing': [0]
        })

        # 3. Vorhersagen
        prediction = model.predict(input_data)[0]

        # 4. Ergebnis anzeigen
        st.success("Berechnung erfolgreich!")
        st.markdown(f"""
        <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px; border-left: 5px solid #007BFF; text-align: center;">
            <h3 style="margin:0; color: #333;">Geschätzte Kaltmiete</h3>
            <h1 style="margin:10px 0; color: #007BFF; font-size: 3.5rem;">{prediction:,.2f} €</h1>
            <p style="color: #666;">für {living_space} m² in {selected_city}</p>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Fehler: Die Model-Datei 'mzyana_lightgbm_model.pkl' wurde nicht gefunden.")
    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")

