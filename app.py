import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. HELFER-KLASSEN (Diese MÜSSEN im Code bleiben für das Modell)
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
# 2. DESIGN & CSS (Blauer Button & Schräges Feld)
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    /* Hintergrund leicht grau */
    .stApp { background-color: #F9FAFB; }
    
    /* Button Style: Blau mit weißem Text */
    div.stButton > button {
        background-color: #007BFF !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        height: 50px !important;
        width: 100%;
        border: none;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #0056b3 !important;
        transform: translateY(-2px);
    }

    /* Das schräge Feld (Wohnungstyp) - Zielt auf das 4. Selectbox Element */
    div[data-testid="stSelectbox"]:nth-of-type(4) > div > div {
        transform: skewX(-10deg);
        border: 2px solid #007BFF !important;
        border-radius: 6px;
        background-color: white;
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

# Session State initialisieren
if 'bl' not in st.session_state: st.session_state.bl = "Bayern"
if 'st' not in st.session_state: st.session_state.st = "München"
if 'plz' not in st.session_state: st.session_state.plz = "80331"

def sync_plz():
    """Wenn PLZ getippt wird -> Stadt/Land updaten"""
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

# --- STANDORT (Live-Update) ---
st.subheader("1. Standort")
# PLZ Input außerhalb des Forms für sofortiges Feedback
plz = st.text_input("Postleitzahl (PLZ)", value=st.session_state.plz, key="plz_in", on_change=sync_plz)

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

# --- HAUPTFORMULAR ---
with st.form("main_form"):
    st.subheader("2. Details")
    c1, c2 = st.columns(2)
    with c1:
        flaeche = st.number_input("Wohnfläche (m²)", 10, 500, 60)
        zimmer = st.number_input("Zimmer", 1.0, 10.0, 2.0, step=0.5)
    with c2:
        etage = st.number_input("Etage (0=EG)", -1, 30, 1)
        baujahr = st.number_input("Baujahr", 1900, 2026, 2000)

    st.subheader("3. Ausstattung")
    # Mappings (Deutsch -> Modell Englisch)
    HEIZ = {"Zentral": "central_heating", "Gas": "gas_heating", "Fernwärme": "district_heating", "Fußboden": "floor_heating", "Öl": "oil_heating"}
    ZUST = {"Gepflegt": "well_kept", "Modernisiert": "modernized", "Erstbezug": "first_time_use", "Neuwertig": "mint_condition", "Saniert": "refurbished"}
    TYP = {"Etagenwohnung": "apartment", "Dachgeschoss": "roof_storey", "Maisonette": "maisonette", "Loft": "loft", "Penthouse": "penthouse"}
    
    col_h, col_t = st.columns(2)
    with col_h: h_wahl = st.selectbox("Heizung", list(HEIZ.keys()))
    with col_t: t_wahl = st.selectbox("Wohnungstyp", list(TYP.keys())) # Dies ist das schräge Feld

    st.subheader("4. Extras")
    chk1, chk2, chk3 = st.columns(3)
    with chk1: balk = st.checkbox("Balkon")
    with chk2: kuech = st.checkbox("Einbauküche")
    with chk3: aufz = st.checkbox("Aufzug")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("JETZT PREIS BERECHNEN 🚀")

# ==============================================================================
# 5. VORHERSAGE LOGIK (Robust & Sicher)
# ==============================================================================
if submit:
    # Wir suchen erst nach deinem hochgeladenen Namen, dann nach Alternativen
    file_options = ['mzyana_lightgbm_model.pkl', 'final_model.pkl']
    model = None
    
    for f in file_options:
        if os.path.exists(f):
            try:
                model = joblib.load(f)
                break
            except Exception as e:
                st.warning(f"Konnte {f} nicht laden: {e}")
                continue
            
    if model is None:
        st.error("❌ FEHLER: Keine Modell-Datei gefunden! Bitte stelle sicher, dass 'mzyana_lightgbm_model.pkl' hochgeladen ist.")
        st.info(f"Dateien im aktuellen Ordner: {os.listdir()}")
    else:
        try:
            # Daten für das Modell vorbereiten
            df = pd.DataFrame({
                'date': [pd.to_datetime(datetime.now())], 
                'livingSpace': [float(flaeche)],
                'noRooms': [float(zimmer)], 
                'floor': [float(etage)], 
                'regio1': [sel_bl],
                'regio2': [sel_st], 
                'heatingType': [HEIZ.get(h_wahl, "central_heating")], 
                'condition': [ZUST.get("Gepflegt")], 
                'interiorQual': ["normal"], 
                'typeOfFlat': [TYP.get(t_wahl, "apartment")], 
                'geo_plz': [str(plz)],
                'balcony': [balk], 
                'lift': [aufz], 
                'hasKitchen': [kuech], 
                'garden': [False],
                'cellar': [True], 
                'yearConstructed': [float(baujahr)],
                # Dummy-Spalten für fehlende Werte (Standard 0)
                'condition_was_missing': [0], 'interiorQual_was_missing': [0],
                'heatingType_was_missing': [0], 'yearConstructed_was_missing': [0]
            })

            preis = model.predict(df)[0]
            
            st.success("Berechnung erfolgreich!")
            st.markdown(f"""
            <div style="background-color: #E3F2FD; padding: 20px; border-radius: 12px; border: 2px solid #007BFF; text-align: center; margin-top: 10px;">
                <h3 style="color: #444; margin: 0;">Geschätzte Kaltmiete</h3>
                <h1 style="color: #007BFF; font-size: 3.5rem; margin: 10px 0;">{preis:,.2f} €</h1>
                <p style="color: #666; font-size: 1.1rem;">für {flaeche} m² in {sel_st}</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Ein Fehler ist bei der Berechnung aufgetreten: {e}")
            st.error("Tipp: Prüfe deine requirements.txt (muss 'lightgbm' enthalten).")
