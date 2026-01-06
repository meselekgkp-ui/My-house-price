import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. NOTWENDIGE KLASSEN (CUSTOM TRANSFORMERS)
# ==============================================================================

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col):
        self.date_col = date_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['post_year'] = X[self.date_col].dt.year
        X['post_month'] = X[self.date_col].dt.month
        return X.drop(columns=[self.date_col])

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col
        self.group_medians = {}
        self.global_median = 0
    def fit(self, X, y=None):
        self.global_median = X[self.target_col].median()
        self.group_medians = X.groupby(self.group_col)[self.target_col].median().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X.apply(
            lambda row: self.group_medians.get(row[self.group_col], self.global_median)
            if pd.isna(row[self.target_col]) else row[self.target_col], axis=1
        )
        return X

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col
        self.mappings = {}
        self.global_mean = 0
    def fit(self, X, y=None):
        self.global_mean = X[self.target_col].mean()
        self.mappings = X.groupby(self.group_col)[self.target_col].mean().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.group_col + '_encoded'] = X[self.group_col].map(self.mappings).fillna(self.global_mean)
        return X.drop(columns=[self.group_col])

# ==============================================================================
# 2. REGIONALE DATENSTRUKTUR (AUSWAHL-LOGIK)
# ==============================================================================

# Beispielhafte Datenstruktur für die Hierarchie. 
# Für die finale Version können Sie diese Liste mit allen Werten aus Ihrem Datensatz füllen.
REGION_DATA = {
    "Bayern": {
        "Deggendorf": ["94469", "94447"],
        "Muenchen": ["80331", "80333", "80801"],
        "Passau": ["94032", "94034", "94036"]
    },
    "Berlin": {
        "Berlin": ["10115", "10117", "10435", "10785", "14195"]
    },
    "Nordrhein_Westfalen": {
        "Koeln": ["50667", "50668", "50733"],
        "Duesseldorf": ["40210", "40212", "40476"]
    }
}

# ==============================================================================
# 3. SEITENKONFIGURATION & STYLING (PROFESSIONELLER LOOK)
# ==============================================================================

st.set_page_config(
    page_title="Mietpreis-Expertensystem",
    layout="wide"
)

# Behörden-Design (Dunkelblau/Weiß)
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    .stHeader { color: #003366; font-family: 'Arial', sans-serif; }
    .stButton>button { 
        width: 100%; 
        background-color: #003366; 
        color: white; 
        border-radius: 0px;
        height: 3.5em;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover { background-color: #002244; color: white; }
    h1 { color: #003366; border-bottom: 3px solid #003366; padding-bottom: 10px; font-weight: bold; }
    h3 { color: #003366; border-left: 5px solid #003366; padding-left: 10px; margin-top: 30px; }
    .stSelectbox, .stNumberInput { font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. DATENABFRAGE (HIERARCHISCHES FORMULAR)
# ==============================================================================

st.title("Mietpreis-Analysesystem für Wohnraum")
st.write("Wissenschaftliche Prognose der Gesamtmiete basierend auf dem aktuellen LightGBM-Modell.")

with st.form("experten_form"):
    
    # --- BLOCK 1: Regionale Auswahl (Die Hierarchie) ---
    st.subheader("Regionale Identifikation")
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        selected_regio1 = st.selectbox("Bundesland", sorted(list(REGION_DATA.keys())))
    
    with col_r2:
        available_cities = sorted(list(REGION_DATA[selected_regio1].keys()))
        selected_regio2 = st.selectbox("Stadt / Kreis", available_cities)
        
    with col_r3:
        available_plz = sorted(REGION_DATA[selected_regio1][selected_regio2])
        selected_geo_plz = st.selectbox("Postleitzahl", available_plz)

    # --- BLOCK 2: Objektdaten ---
    st.subheader("Objektspezifische Merkmale")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        living_space = st.number_input("Wohnfläche (m²)", 10.0, 500.0, 75.0)
    with col2:
        no_rooms = st.number_input("Zimmeranzahl", 1.0, 15.0, 3.0)
    with col3:
        floor = st.number_input("Etage", 0, 20, 1)
    with col4:
        yearConstructed = st.number_input("Baujahr (0 = Unbekannt)", 0, 2025, 1990)

    # --- BLOCK 3: Ausstattung & Qualität ---
    st.subheader("Qualitative Merkmale")
    col5, col6, col7 = st.columns(3)
    with col5:
        heatingType = st.selectbox("Heizungsart", ["central_heating", "district_heating", "gas_heating", "heat_pump", "oil_heating"])
    with col6:
        condition = st.selectbox("Zustand", ["first_time_use", "well_kept", "refurbished", "fully_renovated", "modernized"])
    with col7:
        interiorQual = st.selectbox("Ausstattung", ["normal", "sophisticated", "luxury", "simple"])
    
    typeOfFlat = st.selectbox("Wohnungstyp", ["apartment", "roof_storey", "ground_floor", "maisonette", "penthouse"])

    # --- BLOCK 4: Ausstattung (Checkboxen) ---
    st.subheader("Zusatzausstattung")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        balcony = st.checkbox("Balkon / Terrasse")
    with c2:
        lift = st.checkbox("Aufzug")
    with c3:
        hasKitchen = st.checkbox("Einbauküche")
    with c4:
        garden = st.checkbox("Garten")
    with c5:
        cellar = st.checkbox("Keller")

    date_val = st.date_input("Stichtag der Wertermittlung", datetime.now())

    submit_button = st.form_submit_button("Mietpreis-Analyse starten")

# ==============================================================================
# 5. PROGNOSE-BERECHNUNG
# ==============================================================================

if submit_button:
    try:
        model = joblib.load('mzyana_model_final.pkl')
        
        # Logik für fehlende Werte
        y_missing = 1 if yearConstructed == 0 else 0
        year_val = np.nan if yearConstructed == 0 else yearConstructed

        # DataFrame Erstellung (Reihenfolge muss dem Training entsprechen)
        input_df = pd.DataFrame({
            'date': [pd.to_datetime(date_val)],
            'livingSpace': [float(living_space)],
            'noRooms': [float(no_rooms)],
            'floor': [float(floor)],
            'regio1': [selected_regio1],
            'regio2': [selected_regio2],
            'heatingType': [heatingType],
            'condition': [condition],
            'interiorQual': [interiorQual],
            'typeOfFlat': [typeOfFlat],
            'geo_plz': [str(selected_geo_plz)],
            'balcony': [bool(balcony)],
            'lift': [bool(lift)],
            'hasKitchen': [bool(hasKitchen)],
            'garden': [bool(garden)],
            'cellar': [bool(cellar)],
            'yearConstructed': [year_val],
            'condition_was_missing': [0],
            'interiorQual_was_missing': [0],
            'heatingType_was_missing': [0],
            'yearConstructed_was_missing': [y_missing]
        })

        prediction = model.predict(input_df)[0]

        # Ergebnisanzeige
        st.markdown("---")
        res_col_l, res_col_r = st.columns([2, 1])
        with res_col_l:
            st.subheader("Berechnetes Mietpreisniveau")
            st.markdown(f"**Geschätzte monatliche Gesamtmiete:**")
            st.title(f"{prediction:,.2f} EUR")
        with res_col_r:
            st.subheader("Analyse-Details")
            st.write(f"Modell-Konfidenz: Hoch")
            st.write(f"Basis: LightGBM Gradient Boosting")

    except Exception as e:
        st.error(f"Systemfehler bei der Datenverarbeitung: {str(e)}")

# Footer
st.markdown("---")
st.write("Akademisches Projekt zur datengestützten Immobilienbewertung | Prof. Wahl")
