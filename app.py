import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
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



st.set_page_config(page_title="Mzyana Rent AI", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 Mzyana: KI-Mietpreisrechner")
st.write("Geben Sie die Details der Wohnung ein, um eine KI-basierte Preisschätzung zu erhalten.")
st.info("Dieses Modell basiert على LightGBM مع دقة R² تزيد عن 90%.")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        living_space = st.number_input("Wohnfläche (m²)", min_value=10.0, max_value=500.0, value=75.0, step=1.0)
        no_rooms = st.number_input("Anzahl der Zimmer", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
        date_input = st.date_input("Datum der Veröffentlichung")

    with col2:
        geo_plz = st.text_input("Postleitzahl (PLZ)", "10115")
        regio2 = st.text_input("Stadt (z.B. Berlin)", "Berlin")

st.markdown("---")

# ==============================================================================
# ==============================================================================

if st.button("Mietpreis jetzt schätzen"):
    try:
        model = joblib.load('mzyana_model_final.pkl')
        
        input_data = pd.DataFrame({
            'date': [pd.to_datetime(date_input)],
            'livingSpace': [living_space],
            'noRooms': [no_rooms],
            'geo_plz': [geo_plz],
            'regio2': [regio2],
            'yearConstructed': [np.nan]  
        })

        prediction = model.predict(input_data)[0]

        
        st.balloons()
        st.markdown(f"### 💶 Der geschätzte Mietpreis beträgt:")
        st.header(f"{prediction:,.2f} €")
        
        st.success("Berechnung erfolgreich abgeschlossen!")

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")
        st.warning("Stellen Sie sicher, dass 'mzyana_model_final.pkl' im selben Ordner liegt.")

st.markdown("---")

st.caption("Entwickelt von [Ayman] als Teil des ML Projekt - Betreut von Prof. Wahl.")
