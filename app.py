import streamlit as st
import json
import pandas as pd
import joblib
from datetime import datetime

# ==============================================================================
# 1. KONFIGURATION & DATEN LADEN
# ==============================================================================
st.set_page_config(page_title="Mzyana AI", layout="centered")

@st.cache_data
def get_data():
    try:
        with open('geo_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Erstelle eine Rückwärts-Suche für PLZ -> (Stadt, Bundesland)
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
# 2. SESSION STATE (Die "Logik-Zentrale")
# ==============================================================================
# Wir speichern hier, was aktuell ausgewählt ist
if 'sel_state' not in st.session_state: st.session_state.sel_state = "Bayern"
if 'sel_city' not in st.session_state: st.session_state.sel_city = "München"
if 'sel_plz' not in st.session_state: st.session_state.sel_plz = "80331"

def sync_from_plz():
    """Funktion: Wenn PLZ sich ändert, update Stadt und Bundesland"""
    p = st.session_state.input_plz
    if p in PLZ_LOOKUP:
        stadt, bl = PLZ_LOOKUP[p]
        st.session_state.sel_state = bl
        st.session_state.sel_city = stadt
        st.session_state.sel_plz = p

# ==============================================================================
# 3. DESIGN (CSS) - Schräg & Heller Button
# ==============================================================================
st.markdown("""
    <style>
    /* Button mit hellem Text */
    .stButton>button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 100%;
    }
    
    /* DESIGN WUNSCH: Wohnungstyp ein bisschen schräg */
    div[data-testid="stSelectbox"]:nth-of-type(4) {
        transform: skewX(-5deg);
        border: 1px solid #007BFF;
        padding: 5px;
        border-radius: 5px;
    }
    
    /* Labels dunkel für bessere Sichtbarkeit */
    label { color: #31333F !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. DAS FORMULAR
# ==============================================================================
st.title("Intelligente Immobiliensuche")

with st.container():
    # A. POSTLEITZAHL (Der Master-Trigger)
    plz_input = st.text_input(
        "Postleitzahl eingeben (tippen zum Synchronisieren)", 
        value=st.session_state.sel_plz,
        key="input_plz",
        on_change=sync_from_plz
    )

    # B. BUNDESLAND (Searchable Selectbox)
    all_states = sorted(list(GEO_DATA.keys()))
    try:
        state_index = all_states.index(st.session_state.sel_state)
    except: state_index = 0
    
    selected_state = st.selectbox(
        "Bundesland", 
        all_states, 
        index=state_index, 
        key="state_box"
    )

    # C. STADT (Gefiltert nach Bundesland)
    cities_in_state = sorted(list(GEO_DATA.get(selected_state, {}).keys()))
    try:
        city_index = cities_in_state.index(st.session_state.sel_city)
    except: city_index = 0
    
    selected_city = st.selectbox(
        "Stadt (Nur aus gewähltem Bundesland)", 
        cities_in_state, 
        index=city_index
    )

    # D. WOHNUNGSTYP (Wird durch CSS schräg gestellt)
    st.selectbox("Wohnungstyp", ["Etagenwohnung", "Dachgeschoss", "Loft", "Maisonette"])

    # E. GRENZEN (Zimmer & Fläche)
    col1, col2 = st.columns(2)
    with col1:
        zimmer = st.number_input("Zimmer Anzahl (Min)", 1.0, 10.0, 2.0, step=0.5)
    with col2:
        flaeche = st.number_input("Wohnfläche m² (Min)", 10, 500, 60)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("JETZT SUCHEN"):
        st.success(f"Suche läuft in {selected_city} ({plz_input})...")

# Update der Session bei manueller Wahl
st.session_state.sel_state = selected_state
st.session_state.sel_city = selected_city
