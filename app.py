import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. CUSTOM CLASSES (NICHT ÄNDERN - PFLICHT FÜR MODELL)
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
# 2. DESIGN & CSS (FIX FÜR UNSICHTBARE FELDER)
# ==============================================================================

st.set_page_config(page_title="Immobilienwert-Rechner", layout="wide", page_icon="🏢")

# High-Contrast CSS: Erzwingt schwarze Schrift auf weißem Grund für alle Felder
st.markdown("""
    <style>
    /* Hintergrund der gesamten App */
    .stApp {
        background-color: #F0F2F6;
    }
    
    /* Überschriften */
    h1, h2, h3 {
        color: #0E1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* FIX: Eingabefelder sichtbar machen (Schwarze Schrift, Weißer Hintergrund, Grauer Rand) */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        color: #31333F !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D6D6D6 !important;
    }
    
    /* Dropdown-Menü Optionen */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    
    /* Button Design */
    .stButton>button {
        background-color: #004E8A;
        color: white;
        border-radius: 4px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #003B6D;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Ergebnis-Box */
    .metric-card {
        background-color: #FFFFFF;
        border-left: 5px solid #004E8A;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. DATENBANK LADEN & MAPPINGS
# ==============================================================================

# Datenbank laden
try:
    with open('geo_data.json', 'r', encoding='utf-8') as f:
        GEO_DATA = json.load(f)
except FileNotFoundError:
    st.error("⚠️ Datenbank 'geo_data.json' fehlt! Bitte laden Sie die Datei hoch.")
    GEO_DATA = {"Berlin": {"Berlin": ["10115"]}} # Fallback, damit App nicht abstürzt

# Mappings (Deutsch -> Modell-Englisch)
HEATING_MAP = {
    "Zentralheizung": "central_heating", "Fernwärme": "district_heating",
    "Gas-Heizung": "gas_heating", "Etagenheizung": "self_contained_central_heating",
    "Fußbodenheizung": "floor_heating", "Ölheizung": "oil_heating",
    "Wärmepumpe": "heat_pump", "Holzpelletheizung": "wood_pellet_heating",
    "Andere / Unbekannt": "central_heating" # Fallback
}

CONDITION_MAP = {
    "Gepflegt": "well_kept", "Erstbezug": "first_time_use", 
    "Saniert": "refurbished", "Vollständig renoviert": "fully_renovated",
    "Neuwertig": "mint_condition", "Modernisiert": "modernized",
    "Erstbezug nach Sanierung": "first_time_use_after_refurbishment", 
    "Renovierungsbedürftig / Andere": "negotiable"
}

TYPE_MAP = {
    "Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey",
    "Erdgeschoss": "ground_floor", "Maisonette": "maisonette",
    "Hochparterre": "raised_ground_floor", "Penthouse": "penthouse",
    "Souterrain": "half_basement", "Andere": "apartment"
}

QUAL_MAP = {"Normal": "normal", "Gehoben": "sophisticated", "Luxus": "luxury", "Einfach": "simple"}

# ==============================================================================
# 4. DAS INTERFACE (DIE APP)
# ==============================================================================

st.title("Qualifizierter Mietpreis-Rechner")
st.markdown("---")

with st.form("main_form"):
    
    # --- SEKTION 1: LAGE (INTELLIGENTE SUCHE) ---
    st.subheader("1. Lage der Immobilie")
    col_loc1, col_loc2, col_loc3 = st.columns(3)
    
    with col_loc1:
        # Schritt 1: Bundesland wählen
        states = sorted(list(GEO_DATA.keys()))
        selected_state = st.selectbox("Bundesland", states, index=0)
    
    with col_loc2:
        # Schritt 2: Städte basierend auf Bundesland (Mit Suchfunktion!)
        cities = sorted(list(GEO_DATA[selected_state].keys()))
        # Wenn 'Berlin' oder 'Hamburg' gewählt, ist die Stadt oft gleich dem Land
        default_idx = 0
        selected_city = st.selectbox("Stadt / Landkreis (Tippen zum Suchen)", cities, index=default_idx)
        
    with col_loc3:
        # Schritt 3: PLZ basierend auf Stadt
        plzs = sorted(GEO_DATA[selected_state][selected_city])
        selected_plz = st.selectbox("Postleitzahl", plzs)

    st.info(f"📍 Standort gewählt: {selected_plz} {selected_city}, {selected_state}")

    # --- SEKTION 2: OBJEKTDATEN ---
    st.subheader("2. Gebäudedaten")
    col_obj1, col_obj2, col_obj3, col_obj4 = st.columns(4)
    
    with col_obj1:
        living_space = st.number_input("Wohnfläche (m²)", min_value=10, max_value=600, value=75, step=1)
    with col_obj2:
        rooms = st.number_input("Zimmer", min_value=1, max_value=15, value=3, step=1)
    with col_obj3:
        floor = st.number_input("Etage", min_value=0, max_value=50, value=1, step=1)
    with col_obj4:
        year = st.number_input("Baujahr (0 wenn unbekannt)", min_value=0, max_value=2025, value=1995, step=1)

    # --- SEKTION 3: QUALITÄT ---
    st.subheader("3. Ausstattung & Zustand")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    
    with col_q1: heating = st.selectbox("Heizung", list(HEATING_MAP.keys()))
    with col_q2: condition = st.selectbox("Zustand", list(CONDITION_MAP.keys()))
    with col_q3: quality = st.selectbox("Qualität", list(QUAL_MAP.keys()))
    with col_q4: flat_type = st.selectbox("Typ", list(TYPE_MAP.keys()))

    # --- SEKTION 4: EXTRAS (CHECKBOXEN) ---
    st.markdown("##### Zusätzliche Merkmale")
    col_ex1, col_ex2, col_ex3, col_ex4, col_ex5 = st.columns(5)
    with col_ex1: has_balcony = st.checkbox("Balkon/Terrasse")
    with col_ex2: has_lift = st.checkbox("Aufzug")
    with col_ex3: has_kitchen = st.checkbox("Einbauküche")
    with col_ex4: has_garden = st.checkbox("Garten")
    with col_ex5: has_cellar = st.checkbox("Keller")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Mietpreis berechnen")

# ==============================================================================
# 5. BERECHNUNG & ERGEBNIS
# ==============================================================================

if submit:
    try:
        # Modell laden
        model = joblib.load('mzyana_model_final.pkl')
        
        # DataFrame bauen (Exakt wie beim Training!)
        input_data = pd.DataFrame({
            'date': [pd.Timestamp.now()],
            'livingSpace': [float(living_space)],
            'noRooms': [float(rooms)],
            'floor': [float(floor)],
            'regio1': [selected_state],
            'regio2': [selected_city],
            'heatingType': [HEATING_MAP[heating]],
            'condition': [CONDITION_MAP[condition]],
            'interiorQual': [QUAL_MAP[quality]],
            'typeOfFlat': [TYPE_MAP[flat_type]],
            'geo_plz': [str(selected_plz)],
            'balcony': [has_balcony],
            'lift': [has_lift],
            'hasKitchen': [has_kitchen],
            'garden': [has_garden],
            'cellar': [has_cellar],
            'yearConstructed': [np.nan if year == 0 else float(year)],
            # Missing Flags setzen
            'condition_was_missing': [0],
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0],
            'yearConstructed_was_missing': [1 if year == 0 else 0]
        })

        # Vorhersage
        prediction = model.predict(input_data)[0]

        # Professionelle Anzeige
        st.markdown("---")
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-bottom: 0;">Geschätzte Gesamtmiete</h3>
            <h1 style="color: #004E8A; font-size: 3.5em; margin: 10px 0;">{prediction:,.2f} €</h1>
            <p style="color: grey;">Dieser Wert basiert auf einer KI-Analyse vergleichbarer Objekte in {selected_city}.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error("Fehler: Modelldatei oder Datenbank nicht gefunden. Bitte GitHub prüfen.")
    except Exception as e:
        st.error(f"Ein technischer Fehler ist aufgetreten: {e}")

# Footer
st.markdown("---")
st.caption("Masterprojekt Data Science | Prof. Wahl | Version 2.0 (Stable)")
