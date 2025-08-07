# Estymator czasu półmaratonu

Aplikacja Streamlit do estymacji czasu przebiegnięcia półmaratonu na podstawie danych użytkownika oraz wytrenowanego modelu ML.

## Funkcje aplikacji

### 1. Pobieranie modelu z Digital Ocean Spaces
Model ML (`Gradient_Regressor_pipeline.pkl`) jest automatycznie pobierany z Digital Ocean Spaces przy starcie aplikacji. Dzięki temu zawsze używasz najnowszej wersji modelu.

### 2. Interfejs użytkownika
- **Nagłówek z ikonką biegacza** – aplikacja jest przyjazna wizualnie.
- **Pole tekstowe** – użytkownik podaje swobodnie płeć, wiek oraz czas na 5 km (np. „Jestem mężczyzną, mam 30 lat i przebiegłem 5 km w czasie 22:30”).

### 3. Wyodrębnianie danych z tekstu
- Aplikacja korzysta z LLM (OpenAI) do wyłuskania z tekstu trzech kluczowych informacji:
  - płeć (zakodowana jako liczba: 0 – kobieta, 1 – mężczyzna)
  - wiek (liczba całkowita)
  - czas na 5 km (w różnych formatach, np. `mm:ss`, `minuty sekundy`, liczba minut)

- Jeśli brakuje którejś z danych, aplikacja informuje użytkownika, jakie dane są wymagane.

### 4. Przetwarzanie i prezentacja danych
- Wyodrębnione dane są prezentowane w formie tabeli (DataFrame) dla lepszej czytelności.
- Czas na 5 km jest automatycznie konwertowany do formatu wymaganego przez model (min/km).

### 5. Predykcja czasu półmaratonu
- Dane wejściowe są przekazywane do wytrenowanego modelu ML (PyCaret).
- Model zwraca przewidywany czas przebiegnięcia półmaratonu w sekundach.
- Wynik jest prezentowany w czytelnym formacie (godziny, minuty, sekundy).

### 6. Informacje dodatkowe
- Aplikacja pokazuje również tempo na 5 km, które zostało użyte do predykcji.

### 7. Obsługa błędów
- Jeśli dane są niepoprawne lub nie można wykonać predykcji, użytkownik otrzymuje czytelną informację o błędzie.

### 8. Zbieranie metryk LLM
- Integracja z Langfuse pozwala na zbieranie metryk skuteczności działania LLM podczas wyodrębniania danych z tekstu użytkownika.

---

## Jak uruchomić aplikację

1. Upewnij się, że masz zainstalowane wszystkie wymagane pakiety (`requirements.txt` lub `environment.yml`).
2. W pliku `.env` umieść swoje klucze dostępowe do Digital Ocean Spaces oraz OpenAI.
3. Uruchom aplikację poleceniem:
   ```
   streamlit run app.py
   ```

---

## Wymagania

- Python 3.8+
- Streamlit
- PyCaret
- Pandas
- Boto3
- Langfuse
- OpenAI
- Python-dotenv

---

## Przykład użycia

W polu tekstowym wpisz:
```
Jestem Michał, jestem mężczyzną, mam 36 lat, a mój czas na 5km to 30 min
```
Aplikacja wyodrębni dane, wyświetli je w tabeli i poda przewidywany czas półmaratonu.

---

## Licencja

Projekt edukacyjny – do użytku