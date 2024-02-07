# Projekt Data Analysis

## Beschreibung
Dieses Projekt analysiert Online-Beschwerdedaten des Berliner Ordnungsamts, um häufige Themen und Beschwerden mittels Natural Language Processing (NLP) Techniken zu identifizieren. Es verwendet Methoden wie Textvorverarbeitung, Bag-of-Words (BoW), TF-IDF, Latent Semantic Analysis (LSA) und Latent Dirichlet Allocation (LDA) zur Extraktion und Analyse der Themen.

## System-Requirements
- Python Installation (https://www.python.org/downloads/)
- Tool zum herunterladen des Repository (Bspw. Git-Scm: https://git-scm.com/)
- Um eine Conda-Umgebung zu erstellen und zu nutzen, müssen Sie sicherstellen, dass Conda auf Ihrem System installiert ist.
  - Bspw.: Anaconda- oder Miniconda-Distributionen (https://docs.anaconda.com/free/miniconda/).

## Installation
Um dieses Projekt auszuführen, müssen Sie zuerst eine Conda-Umgebung einrichten und die erforderlichen Pakete installieren. Stellen Sie sicher, dass Conda auf Ihrem System installiert ist, und folgen Sie den untenstehenden Schritten:

Conda Terminal öffnen

**1. Klonen des Repositorys:**

git clone https://github.com/MarcoSer1/ProjektDataAnalysis

cd <Repository_Pfad>

**2. Erstellen der Conda-Umgebung:**

conda env create -f environment.yml

**3. Aktivieren der Umgebung:**

conda activate ProjektDataAnalysis

**4. Ausführen des Programms:**

python pythoncode.py

## Funktion des Codes:

- **preprocess_complex_words(text):** Bereinigt den Text, indem spezifische Muster durch angepasste Formen ersetzt werden.

- **remove_unwanted_characters(token):** Entfernt unerwünschte Zeichen aus den Token, behält aber Buchstaben und Umlaute.

- **clean_text(text):** Führt die gesamte Textvorverarbeitung durch, einschließlich Tokenisierung, Entfernung von Stoppwörtern, Lemmatisierung und weiterer Bereinigung.

- Der Code verarbeitet die Daten schrittweise, um "saubere Texte" zu erzeugen, die dann für die Vektorisierung und anschließende Themenextraktion verwendet werden.

## Verwendung des Codes:
Ersetzen Sie <Repository_Name> mit den tatsächlichen Werten des Projekts. Passen Sie außerdem den Namen der Conda-Umgebung (mein_projekt) und den Skriptnamen (pythoncode.py) entsprechend an.
