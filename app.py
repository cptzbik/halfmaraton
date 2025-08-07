import json
import pandas as pd
from pycaret.regression import load_model
from pathlib import Path
import boto3
import os
import streamlit as st
from dotenv import dotenv_values, load_dotenv

from langfuse.decorators import observe
from langfuse.openai import OpenAI

env = dotenv_values(".env")
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = env["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = env["AWS_SECRET_ACCESS_KEY"]
os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"] = env["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = env["LANGFUSE_SECRET_KEY"]

openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

# Konfiguracja klienta S3 (Digital Ocean Spaces)
session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    region_name='fra1', 
    endpoint_url='https://maratonapp.fra1.digitaloceanspaces.com',
    aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"]
)

# Pobierz model z Spaces do pliku tymczasowego
MODEL_BUCKET = "maratonapp"
MODEL_KEY = "Gradient_Regressor_pipeline.pkl"
LOCAL_MODEL_PATH = "Gradient_Regressor_pipeline.pkl"

s3.download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

# Ładuj model PyCaret
regressor = load_model("Gradient_Regressor_pipeline")

@observe()
def extract_structured_data(free_text: str):
    system_prompt = (
        "Jesteś parsującym asystentem. Ze swobodnego opisu użytkownika wyciągnij dokładnie trzy pola:\n"
        "1. płeć_encoded: 0 jeśli kobieta, 1 jeśli mężczyzna (jeśli ujmuje słownie: kobieta, mężczyzna, pani, pan etc.)\n"
        "2. wiek: liczba całkowita\n"
        "3. 5_km_tempo: tempo na 5 km w minutach na kilometr (np. 3.5 dla 3 minut 30 sekund na km). Użytkownik może podawać w formacie mm:ss, m:ss, np. '22:30' (4:30 min/km), '23 minuty 10 sekund' (4:37 min/km), '25 minut' (5:00 min/km), '23.5 minut' (4:42 min/km).\n"
        "Odpowiedz tylko w czystym JSON-ie z kluczami: płeć_encoded, wiek, 5 km Tempo. Jeśli czegoś brakuje, nie zgaduj, daj wartość null dla brakującego pola. Jeśli użytkownik poda rok urodzenia oblicz jego wiek, biorąc pod uwagę bieżący rok (teraz mamy 2025).\n"
    )
    user_prompt = free_text.strip()
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    try:
        data = json.loads(response.choices[0].message.content)
    except Exception:
        data = {}
    return data

def convert_time_to_tempo(time_str):
    """Konwertuje czas na 5 km na tempo w minutach na kilometr"""
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_minutes = minutes + seconds / 60
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            total_minutes = hours * 60 + minutes + seconds / 60
    else:
        try:
            total_minutes = float(time_str)
        except:
            return None
    
    # Dla 5 km, tempo = czas / 5
    tempo_per_km = total_minutes / 5
    return tempo_per_km

st.image("https://maratonapp.fra1.digitaloceanspaces.com/maratonapp/Maraton.png")

st.title("🏃 Estymator czasu półmaratonu")
st.header("Wprowadź dane biegacza")

user_input = st.text_area("Przedstaw się, podaj płeć, wiek i czas na 5km:(Np. 'Jestem mężczyzną, mam 30 lat i przebiegłem 5 km w czasie 22:30')",)

if st.button("Estymuj czas półmaratonu"):
    extracted = extract_structured_data(user_input)
    missing = []
    for key in ["płeć_encoded", "wiek", "5 km Tempo"]:
        if key not in extracted or extracted[key] is None:
            missing.append(key)
    if missing:
        st.warning(f"Brakuje następujących danych: {', '.join(missing)}")
    else:
        st.success("Dane wyodrębnione poprawnie:")
        

        try:
            gender_val = int(extracted["płeć_encoded"])
            age_val = int(extracted["wiek"])
            
            # Sprawdź czy 5 km Tempo to już liczba czy string do konwersji
            pace_val = extracted["5 km Tempo"]
            if isinstance(pace_val, str):
                # Jeśli to string, sprawdź czy to czas (zawiera :) czy tempo
                if ":" in pace_val:
                    # To jest czas na 5 km, konwertuj na tempo
                    pace_val = convert_time_to_tempo(pace_val)
                    if pace_val is None:
                        st.error("Nieprawidłowy format czasu na 5 km")
                        st.stop()
                else:
                    # To może być tempo bezpośrednio podane
                    try:
                        pace_val = float(pace_val)
                    except:
                        st.error("Nieprawidłowy format tempa na 5 km")
                        st.stop()
            else:
                pace_val = float(pace_val)

            # Przygotowanie inputu jako DataFrame
            input_df = pd.DataFrame({
                "płeć_encoded": [gender_val],
                "wiek": [age_val],
                "5 km Tempo": [pace_val]
            })

            # Predykcja - model zwraca tylko czas półmaratonu w sekundach
            pred_halfmarathon_seconds = regressor.predict(input_df)[0]

            def format_time(total_seconds):
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                secs = int(total_seconds % 60)
                if hours > 0:
                    return f"{hours}h {minutes}min {secs}sek"
                else:
                    return f"{minutes}min {secs}sek"

            st.info(f"Czas na przebiegnięcie półmaratonu to: {format_time(pred_halfmarathon_seconds)}")
            
            # Dodatkowo pokaż tempo na 5 km które zostało użyte
            st.info(f"Prędkość na 5 km: {pace_val:.2f} min/km")
            
        except Exception as e:
            st.error(f"Nie udało się przetworzyć danych lub wykonać predykcji: {str(e)}")
            st.warning("Nie udało się przetworzyć danych lub wykonać predykcji.")