# Chain-of-Translation Prompting (CoTR) für LRLs

Dieses Repository enthält die Implementierung und experimentellen Ergebnisse für **CoTR**, eine Prompting-Strategie, die Übersetzungsketten nutzt, um die Leistung bei NLP-Aufgaben in LRLs zu verbessern.

## Projektübersicht

CoTR adressiert die Herausforderung der schlechten Leistung von MLLMs bei LRLs durch einen übersetzungsbasierten Reasoning-Ansatz. Anstatt Modelle direkt in LRLs zu prompten, übersetzt CoTR Eingaben ins Englische, führt die Aufgabe auf Englisch aus und übersetzt dann die Ergebnisse zurück in die ursprüngliche Sprache.

### Hauptmerkmale

- **Multi-Task-Unterstützung**: Sentiment-Analyse, Textklassifikation, Named Entity Recognition (NER), Natural Language Inference (NLI) und Question Answering (QA)
- **Multi-Modell-Evaluierung**: CohereLabs/aya-23-8B und Qwen/Qwen2.5-7B-Instruct
- **Multi-Sprachen-Abdeckung**: Englisch, Swahili, Hausa, Finnisch, Portugiesisch (MZ), Französisch, Urdu
- **Duale Pipeline-Architektur**: Single-Prompt und Multi-Prompt CoTR-Ansätze
- **Umfassende Evaluierung**: Accuracy, F1-Scores, Parsing-Erfolgsraten, Laufzeitanalyse und Übersetzungsqualitätsbewertung

## Projektstruktur

```
CoTR_Prompting_low_resource/
├── src/
│   ├── experiments/
│   │   ├── baseline/           # Direkte Prompting-Experimente
│   │   │   ├── classification/
│   │   │   ├── ner/
│   │   │   ├── nli/
│   │   │   ├── qa/
│   │   │   └── sentiment/
│   │   ├── cotr/              # Chain-of-Translation-Experimente
│   │   │   ├── classification/
│   │   │   ├── ner/
│   │   │   ├── nli/
│   │   │   ├── qa/
│   │   │   └── sentiment/
│   │   └── simple_baseline/   # Fixed-Prediction-Baselines
│   └── utils/
│       ├── data_loaders/      # Dataset-Laden-Utilities
│       │   ├── load_afrisenti.py
│       │   ├── load_masakhaner.py
│       │   ├── load_masakhanews.py
│       │   ├── load_tydiqa.py
│       │   └── load_xnli.py
│       └── calculate_average_tokens.py  # Token-Analyse und Visualisierung
├── evaluation/
│   ├── descriptive_statistics/ # Parsing-Erfolg, Laufzeitanalyse, Metriken
│   │   ├── calculate_comprehensive_metrics.py
│   │   ├── runtime_analysis_script.py
│   │   ├── analyze_ner_parsing_success.py
│   │   ├── analyze_sentiment_parsing_success.py
│   │   ├── analyze_nli_parsing_success.py
│   │   └── analyze_classification_parsing.py
│   ├── statistical_analysis/   # Hypothesentests
│   │   └── simplified_statistical_analysis.py
│   ├── translation/           # BLEU-Score-Evaluierung
│   │   └── comprehensive_translation_quality_assessment_fixed.py
│   ├── baseline/             # Baseline-spezifische Evaluierung
│   ├── cotr/                 # CoTR-spezifische Evaluierung
│   ├── sentiment_metrics.py  # Sentiment-spezifische Metriken
│   └── classification_metrics.py  # Klassifikations-spezifische Metriken
├── results/                   # Experimentelle Ergebnisse
│   ├── classification_new/
│   ├── ner_new/
│   ├── nli_new/
│   ├── qa_new/
│   └── sentiment_new/
├── config.py                  # HuggingFace-Token-Konfiguration
└── model_initialization.py    # Modell-Laden-Utilities
```

## Experimenteller Aufbau

### Hardware- und Software-Umgebung

- **Hardware**: Cluster mit 4× NVIDIA A6000 GPUs
- **Software**: Python 3.10 mit Hugging Face Ecosystem
- **Wichtige Bibliotheken**: transformers, datasets, evaluate, pandas, numpy, scikit-learn

### Evaluierte Modelle

1. **CohereLabs/aya-23-8B**: Mehrsprachiges Modell 
2. **Qwen/Qwen2.5-7B-Instruct**: Instruction-tuned Modell 

### Datasets und Sprachen

| Aufgabe | Dataset | Sprachen | Datenaufteilung |
|---------|---------|----------|-----------------|
| **Sentiment-Analyse** | AfriSenti | Hausa (ha), Swahili (sw), Portugiesisch (MZ) (pt) | test |
| **Textklassifikation** | MasakhaNEWS | Englisch (en), Hausa (ha), Swahili (sw) | test |
| **NER** | MasakhaNER | Hausa (hau), Swahili (swa) | test |
| **NLI** | XNLI | Englisch (en), Französisch (fr), Swahili (sw), Urdu (ur) | test |
| **QA** | TyDiQA-GoldP | Englisch (en), Finnisch (fi), Swahili (sw) | validation |

## Setup und Konfiguration

### 1. HuggingFace-Token konfigurieren

```bash
python config.py
# Geben Sie Ihren HuggingFace-Token ein, wenn Sie dazu aufgefordert werden
```

Die `config.py` Datei verwaltet automatisch:
- Token-Speicherung in `~/.cotr_config/huggingface_token.txt`
- Sichere Token-Verwaltung mit Berechtigungen 600
- Automatische Token-Abfrage bei fehlendem Token

### 2. Modell-Laden konfigurieren

```python
# Verwendung von model_initialization.py
from model_initialization import initialize_model

# Modell laden
model, tokenizer = initialize_model("CohereLabs/aya-23-8B")
```

### 3. Dataset-Laden konfigurieren

```python
# Dataset-Loader verwenden
from src.utils.data_loaders.load_afrisenti import load_afrisenti_samples
from src.utils.data_loaders.load_masakhanews import load_masakhanews_samples
from src.utils.data_loaders.load_masakhaner import load_masakhaner_samples
from src.utils.data_loaders.load_xnli import load_xnli_samples
from src.utils.data_loaders.load_tydiqa import load_tydiqa_samples

# Beispiele laden
samples = load_afrisenti_samples('sw', num_samples=80, split='test')
```

## Experimente ausführen

### Voraussetzungen

1. **Abhängigkeiten installieren**:
```bash
pip install transformers datasets evaluate pandas numpy scikit-learn matplotlib seaborn tqdm
```

2. **HuggingFace-Token konfigurieren**:
```bash
python config.py
```

### Baseline-Experimente

#### Sentiment-Analyse
```bash
python src/experiments/baseline/sentiment/run_sentiment_baseline_new.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "ha,sw,pt" \
    --num_samples 80 \
    --split test
```

#### Textklassifikation
```bash
python src/experiments/baseline/classification/run_classification_baseline.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,ha,sw" \
    --num_samples 80 \
    --split test
```

#### Named Entity Recognition
```bash
python src/experiments/baseline/ner/run_ner_baseline.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "hau,swa" \
    --num_samples 80 \
    --split test
```

#### Natural Language Inference
```bash
python src/experiments/baseline/nli/run_nli_baseline.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,fr,sw,ur" \
    --num_samples 80 \
    --split test
```

#### Question Answering
```bash
python src/experiments/baseline/qa/run_qa_baseline.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,fi,sw" \
    --num_samples 80 \
    --split validation
```

### CoTR-Experimente

#### Sentiment-Analyse CoTR
```bash
python src/experiments/cotr/sentiment/run_sentiment_cotr.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "ha,sw,pt" \
    --num_samples 80 \
    --split test \
    --pipeline_types "multi_prompt,single_prompt" \
    --shot_settings "zero_shot,few_shot"
```

#### Textklassifikation CoTR
```bash
python src/experiments/cotr/classification/run_classification_cotr.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,ha,sw" \
    --num_samples 80 \
    --split test \
    --pipeline_types "multi_prompt,single_prompt" \
    --shot_settings "zero_shot,few_shot"
```

#### NER CoTR
```bash
python src/experiments/cotr/ner/run_ner_cotr.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "hau,swa" \
    --num_samples 80 \
    --split test \
    --pipeline_types "multi_prompt,single_prompt" \
    --shot_settings "zero_shot,few_shot"
```

#### NLI CoTR
```bash
python src/experiments/cotr/nli/run_nli_cotr.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,fr,sw,ur" \
    --num_samples 80 \
    --split test \
    --pipeline_types "multi_prompt,single_prompt" \
    --shot_settings "zero_shot,few_shot"
```

#### QA CoTR
```bash
python src/experiments/cotr/qa/run_qa_cotr.py \
    --models "CohereLabs/aya-23-8B,Qwen/Qwen2.5-7B-Instruct" \
    --langs "en,fi,sw" \
    --num_samples 80 \
    --split validation \
    --pipeline_types "multi_prompt,single_prompt" \
    --shot_settings "zero_shot,few_shot"
```

### Simple Baseline-Experimente

Führen Sie Fixed-Prediction-Baselines für den Vergleich aus:

```bash
# Sentiment-Analyse Fixed Baselines
python src/experiments/simple_baseline/sentiment_simple_baselines.py \
    --langs "sw,ha,pt" \
    --num_samples 80 \
    --split test

# Textklassifikation Fixed Baselines
python src/experiments/simple_baseline/classification_simple_baselines.py \
    --langs "en,sw,ha" \
    --num_samples 80 \
    --split test

# NLI Fixed Baselines
python src/experiments/simple_baseline/nli_simple_baselines.py \
    --langs "en,ur,sw,fr" \
    --num_samples 80 \
    --split test
```

## Token-Analyse und Visualisierung

### Durchschnittliche Token-Berechnung

```bash
python src/utils/calculate_average_tokens.py \
    --base_results_dir results \
    --output_dir src/utils/token_analysis
```

Diese Analyse:
- Berechnet durchschnittliche Token-Längen für alle Experimente
- Generiert Visualisierungen der Token-Verteilung
- Erstellt Vergleichsplots zwischen Baseline und CoTR
- Exportiert detaillierte Token-Statistiken

## Evaluierung und Analyse

### Umfassende Metriken-Berechnung

```bash
python evaluation/descriptive_statistics/calculate_comprehensive_metrics.py \
    --base_results_dir results \
    --output_dir evaluation/descriptive_statistics/output
```

Diese Analyse berechnet:
- **Accuracy**: Standard-Klassifikationsgenauigkeit
- **F1-Scores**: Weighted, Macro und Custom F1 für verschiedene Aufgaben
- **Task-spezifische Metriken**: QA F1 (Token-Overlap), NER F1 (Entity-Level)
- **Parsing-Erfolgsraten**: Anteil gültiger Modellausgaben
- **Detaillierte Statistiken**: Per-Klasse-Metriken und Konfusionsmatrizen

### Task-spezifische Metriken

#### Sentiment-Metriken
```python
from evaluation.sentiment_metrics import calculate_sentiment_metrics

# Sentiment-spezifische Evaluierung
metrics = calculate_sentiment_metrics(results_df, labels=['positive', 'negative', 'neutral'])
```

#### Klassifikations-Metriken
```python
from evaluation.classification_metrics import calculate_classification_metrics

# Klassifikations-spezifische Evaluierung
metrics = calculate_classification_metrics(results_df, possible_labels_en=['business', 'entertainment', 'health', 'politics', 'religion', 'sports', 'technology'])
```

### Übersetzungsqualitätsbewertung

```bash
python evaluation/translation/comprehensive_translation_quality_assessment_fixed.py
```

Diese Analyse bewertet:
- **BLEU-Scores**: Forward, Backward und Overall BLEU
- **Übersetzungsqualität**: Pro-Task und Pro-Sprache Analyse
- **NLLB-Baseline**: Vergleich mit professionellem Übersetzer
- **Detaillierte Berichte**: Übersetzungsfehler und Qualitätsmetriken

### Statistische Analyse und Hypothesentests

```bash
python evaluation/statistical_analysis/simplified_statistical_analysis.py
```

Diese Analyse testet:

#### H1: Gesamtleistung CoTR vs Baseline
- Paired t-tests für F1-Score und Accuracy
- Relative Verbesserungsprozente
- Statistische Signifikanz (α=0.05)

#### H2: Übersetzungsqualität vs Leistung
- Pearson-Korrelation zwischen BLEU-Scores und F1-Scores
- Task-Level-Korrelationen
- Übersetzungsqualität als Prädiktor für Leistung

#### H3: Aufgabenkomplexität vs CoTR-Effektivität
- Task-spezifische Analysen
- Komplexitätsbasierte Hypothesen
- Relative Verbesserungen pro Aufgabe

#### Pipeline-Effektivität
- Multi-Prompt vs Single-Prompt Vergleich
- Statistische Signifikanz der Pipeline-Unterschiede
- Effektgrößen für Pipeline-Entscheidungen

### Laufzeitanalyse

```bash
python evaluation/descriptive_statistics/runtime_analysis_script.py \
    --base_results_dir results \
    --output_dir evaluation/descriptive_statistics/runtime_analysis
```

Diese Analyse bewertet:
- **Durchschnittliche Laufzeiten**: Pro Sample und Pro Konfiguration
- **Effizienzvergleiche**: Baseline vs CoTR Laufzeiten
- **Skalierungsanalysen**: Laufzeit vs Sample-Größe
- **Hardware-Nutzung**: GPU- und Memory-Verbrauch

### Parsing-Erfolgsanalyse

```bash
# NER Parsing-Erfolg
python evaluation/descriptive_statistics/analyze_ner_parsing_success.py

# Sentiment Parsing-Erfolg
python evaluation/descriptive_statistics/analyze_sentiment_parsing_success.py

# NLI Parsing-Erfolg
python evaluation/descriptive_statistics/analyze_nli_parsing_success.py

# Klassifikation Parsing-Erfolg
python evaluation/descriptive_statistics/analyze_classification_parsing.py
```

Diese Analysen bewerten:
- **Format-Konformität**: Anteil gültiger Modellausgaben
- **Parsing-Fehler**: Typen und Häufigkeiten von Parsing-Fehlern
- **Task-spezifische Validierung**: Format-Validierung pro Aufgabe
- **Detaillierte Fehleranalysen**: Ursachen und Muster von Parsing-Fehlern

## Experimentelles Design

### CoTR-Pipeline-Architektur

#### Multi-Prompt CoTR
1. **Übersetzungsschritt**: LRL → Englisch
2. **Aufgabenverarbeitung**: NLP-Aufgabe auf Englisch ausführen
3. **Rückübersetzung**: Englisch → LRL (für Ausgabe-Labels)

#### Single-Prompt CoTR
1. **Vereinheitlichte Kette**: Einzelner Prompt mit allen Schritten
2. **Strukturierte Ausgabe**: Modell generiert Zwischen- und Endergebnisse
3. **Fehlerpotential**: Potenzial für fortlaufende Fehler

### Generierungsparameter

#### Baseline-Parameter
- **Temperatur**: 0.3 (moderate Zufälligkeit)
- **Max Tokens**: 10-50 (aufgabenabhängig)

#### CoTR-Parameter
- **Übersetzungsschritt**: Höhere Temperatur (0.35) 
- **Aufgabenverarbeitung**: Niedrigere Temperatur (0.1-0.2) für Präzision
- **Max Tokens**: 200-512 (übersetzungsabhängig)
- **Beam Search**: Verwendet für NER (JSON-Ausgabe)
- **Wiederholungsstrafe**: 1.2 (vs 1.1 Baseline)

### Evaluierungsmetriken

#### Leistungsmetriken
- **Accuracy**: Standard-Klassifikationsgenauigkeit
- **F1-Score**: 
  - Weighted F1 für Klassifikation/NLI/Sentiment
  - Custom Token-Overlap F1 für QA
  - Entity-Level F1 für NER

#### Qualitätsmetriken
- **Parsing-Erfolgsrate**: Anteil gültiger Ausgaben
- **Laufzeitanalyse**: Zeit pro Sample/Konfiguration
- **Übersetzungsqualität**: BLEU-Scores (Forward/Backward/Overall)

#### Statistische Analyse
- **Paired t-tests**: Baseline vs CoTR-Vergleich
- **Effektgrößen**: Relative Verbesserungsprozente
- **Aufgaben-Level-Analyse**: Pro-Aufgabe-Effektivität

## Wichtigste Erkenntnisse

### Gesamtleistung
- **CoTR vs Baseline**: +18.4% F1-Score-Verbesserung (p=0.004)
- **Multi-prompt vs Single-prompt**: +29.4% vs +7.5% Verbesserung
- **Übersetzungsqualität**: 32.7% durchschnittlicher BLEU-Score

### Aufgaben-spezifische Ergebnisse
- **NLI**: +54.7% F1-Verbesserung (beste Aufgabe)
- **Sentiment-Analyse**: +73.7% F1-Verbesserung
- **QA**: -53.4% F1-Verschlechterung (Übersetzungsschwierigkeiten)
- **NER**: -13.7% F1-Verschlechterung
- **Klassifikation**: -8.3% F1-Verschlechterung

### Parsing-Erfolgsraten
- **Baseline**: 69.5% (Klassifikation), 59.8% (NER), 100% (NLI), 90.9% (Sentiment)
- **CoTR**: 50.7% (Klassifikation), 0.0% (NER), 100% (NLI), 99.3% (Sentiment)

### Laufzeitanalyse
- **Baseline**: 0.39-6.68 Sekunden pro Sample
- **CoTR**: 3.00-331.87 Sekunden pro Sample
- **Klassifikation**: Rechenintensivste Aufgabe

## Verwendungsbeispiele

### Schnellstart
```bash
# Sentiment-Analyse-Experimente ausführen
python src/experiments/cotr/sentiment/run_sentiment_cotr.py \
    --models "Qwen/Qwen2.5-7B-Instruct" \
    --langs "sw" \
    --num_samples 20 \
    --test_mode
```

### Benutzerdefinierte Konfiguration
```bash
# Mit benutzerdefinierten Generierungsparametern ausführen
python src/experiments/cotr/classification/run_classification_cotr.py \
    --models "CohereLabs/aya-23-8B" \
    --langs "ha,sw" \
    --num_samples 100 \
    --pipeline_types "multi_prompt" \
    --shot_settings "few_shot" \
    --overwrite_results
```

### Vollständige Analyse-Pipeline
```bash
# 1. Experimente ausführen
python src/experiments/cotr/sentiment/run_sentiment_cotr.py --test_mode

# 2. Umfassende Metriken berechnen
python evaluation/descriptive_statistics/calculate_comprehensive_metrics.py

# 3. Übersetzungsqualität bewerten
python evaluation/translation/comprehensive_translation_quality_assessment_fixed.py

# 4. Statistische Analyse durchführen
python evaluation/statistical_analysis/simplified_statistical_analysis.py

# (5. Token-Analyse und Visualisierung)
python src/utils/calculate_average_tokens.py

# 6. Parsing-Erfolg analysieren
python evaluation/descriptive_statistics/analyze_sentiment_parsing_success.py

# 7. Laufzeitanalyse
python evaluation/descriptive_statistics/runtime_analysis_script.py
```

