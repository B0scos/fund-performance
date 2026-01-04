[Read this in English](#financial-asset-clustering--stability-analysis)

[Leia em Português](#análise-de-clusterização-e-estabilidade-de-ativos-financeiros)

---

## Financial Asset Clustering & Stability Analysis

This project implements an end-to-end machine learning pipeline for segmenting financial assets using unsupervised clustering algorithms. It goes beyond standard performance metrics by introducing a robust evaluation framework that assesses model stability across different data partitions (training, testing, and validation).

The primary goal is to identify meaningful asset clusters and ensure that the characteristics of these clusters are consistent and reliable, which is critical for real-world financial applications.

### Project Conclusion and Key Insights

The analysis demonstrates that selecting a model based solely on a single performance metric, such as the Silhouette Score, can be insufficient. A model that performs well on training data may produce inconsistent or unstable clusters when applied to new, unseen data.

By incorporating the **Wasserstein distance** as a stability score, this project establishes a more holistic evaluation methodology. The key finding is that the optimal model is often a trade-off between high performance and high stability. This dual-metric approach ensures that the identified asset clusters (e.g., "high-return, low-risk") are not artifacts of the training set but represent genuinely distinct and reliable groupings. This stability is paramount for building trust and utility in quantitative financial strategies derived from the model's output.

### Key Features

- **Experimentation Pipeline:** A systematic framework for tuning hyperparameters, including the number of clusters, model type (K-Means, GMM), and data preprocessing techniques.
- **Advanced Preprocessing:** Implements multiple strategies such as standard scaling (`scalling`) and dimensionality reduction (`PCA`).
- **Robust Evaluation:**
    - **Performance:** Measured using the Silhouette Score to evaluate cluster density and separation.
    - **Stability:** Measured using the Wasserstein Distance to quantify the statistical similarity of cluster characteristics across train, test, and validation sets.
- **Structured Logging:** All experiment parameters and results are automatically saved to CSV files for comprehensive analysis.
- **Automated Analysis:** An analysis script (`a.py`) processes the results to identify the best-performing model and the most stable model, highlighting the trade-offs between the two.

### Project Structure

- **/src**: Contains the core application code, organized into:
    - `models`: Clustering model wrappers (K-Means, GMM).
    - `pipelines`: Training and evaluation pipeline logic.
    - `process`: Data preprocessing functions.
    - `utils`: Utility functions, including data loading.
- **/main.py**: The main script to execute the complete experiment pipeline.
- **/a.py**: The analysis script to interpret experiment results.
- **/experiment_results.csv**: Output file containing detailed, per-cluster metrics from all runs.

### Setup and Installation

1.  **Prerequisites:**
    - Python 3.9+
    - Git

2.  **Clone the repository:**
    ```bash
    git clone <REPOSITORY_URL>
    cd <PROJECT_DIRECTORY>
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the Experiment Pipeline:**
    Execute the main script to run all configured experiments. This will train the models, evaluate them, and save the results to `experiment_results.csv`.
    ```bash
    python main.py
    ```

2.  **Analyze the Results:**
    Run the analysis script to identify the best-performing and most stable models.
    ```bash
    python a.py
    ```

---

## Análise de Clusterização e Estabilidade de Ativos Financeiros

Este projeto implementa um pipeline de machine learning de ponta a ponta para a segmentação de ativos financeiros usando algoritmos de clusterização não supervisionados. O projeto vai além das métricas de desempenho padrão, introduzindo uma estrutura de avaliação robusta que mede a estabilidade do modelo em diferentes partições de dados (treino, teste e validação).

O objetivo principal é identificar agrupamentos de ativos significativos e garantir que as características desses clusters sejam consistentes e confiáveis, o que é fundamental para aplicações financeiras no mundo real.

### Conclusão do Projeto e Principais Insights

A análise demonstra que selecionar um modelo com base apenas em uma única métrica de desempenho, como o Silhouette Score, pode ser insuficiente. Um modelo com bom desempenho nos dados de treino pode produzir clusters inconsistentes ou instáveis quando aplicado a dados novos e não vistos.

Ao incorporar a **distância de Wasserstein** como uma pontuação de estabilidade, este projeto estabelece uma metodologia de avaliação mais holística. A principal conclusão é que o modelo ideal é frequentemente um equilíbrio entre alto desempenho e alta estabilidade. Essa abordagem de métrica dupla garante que os clusters de ativos identificados (por exemplo, "alto retorno, baixo risco") não sejam artefatos do conjunto de treino, mas representem agrupamentos genuinamente distintos e confiáveis. Essa estabilidade é primordial para construir confiança e utilidade em estratégias financeiras quantitativas derivadas dos resultados do modelo.

### Principais Funcionalidades

- **Pipeline de Experimentação:** Uma estrutura sistemática para o ajuste de hiperparâmetros, incluindo o número de clusters, tipo de modelo (K-Means, GMM) e técnicas de pré-processamento de dados.
- **Pré-processamento Avançado:** Implementa múltiplas estratégias, como padronização (`scalling`) e redução de dimensionalidade (`PCA`).
- **Avaliação Robusta:**
    - **Desempenho:** Medido usando o Silhouette Score para avaliar a densidade e separação dos clusters.
    - **Estabilidade:** Medida usando a Distância de Wasserstein para quantificar a similaridade estatística das características dos clusters entre os conjuntos de treino, teste e validação.
- **Registro Estruturado:** Todos os parâmetros e resultados dos experimentos são salvos automaticamente em arquivos CSV para uma análise abrangente.
- **Análise Automatizada:** Um script de análise (`a.py`) processa os resultados para identificar o modelo de melhor desempenho e o modelo mais estável, destacando o equilíbrio entre os dois.

### Estrutura do Projeto

- **/src**: Contém o código principal da aplicação, organizado em:
    - `models`: Wrappers para os modelos de clusterização (K-Means, GMM).
    - `pipelines`: Lógica do pipeline de treino e avaliação.
    - `process`: Funções de pré-processamento de dados.
    - `utils`: Funções utilitárias, incluindo o carregamento de dados.
- **/main.py**: O script principal para executar o pipeline completo de experimentos.
- **/a.py**: O script de análise para interpretar os resultados dos experimentos.
- **/experiment_results.csv**: Arquivo de saída contendo métricas detalhadas por cluster de todas as execuções.

### Configuração e Instalação

1.  **Pré-requisitos:**
    - Python 3.9+
    - Git

2.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <DIRETORIO_DO_PROJETO>
    ```

3.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # No Windows
    .venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Utilização

1.  **Execute o Pipeline de Experimentos:**
    Execute o script principal para rodar todos os experimentos configurados. Isso treinará os modelos, os avaliará e salvará os resultados em `experiment_results.csv`.
    ```bash
    python main.py
    ```

2.  **Analise os Resultados:**
    Execute o script de análise para identificar os modelos com melhor desempenho e maior estabilidade.
    ```bash
    python a.py
    ```

##  Features

- **Data Ingestion Module:** Automatically collects and prepares raw data.
- **Processing and Cleaning:** Pipelines to validate, clean, and transform data.
- **Feature Engineering:** Creation and selection of features to optimize model performance.
- **Model Training:** Support for multiple clustering algorithms, such as K-Means and Gaussian Mixture Models (GMM).
- **Modular Structure:** Code organized into reusable components, facilitating maintenance and expansion.

##  Project Structure

The project is organized into the following main directories:

- **`/data_ingestion`**: Module responsible for the initial collection and storage of data. It contains its own logic, CLI, and configurations.
- **`/src`**: Contains the main application code, including processing pipelines, model training, and utilities.
- **`/notebooks`**: Jupyter Notebooks for exploratory analysis, testing, and prototyping.
- **`/main.py`**: Main entry point to orchestrate the project's pipelines.
- **`/requirements.txt`**: List of the project's Python dependencies.

##  Getting Started

Follow the instructions below to set up and run the project in your local environment.

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone <REPOSITORY_URL>
    cd <PROJECT_NAME>
    ```

2.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Usage

The project execution is divided into two main steps: data ingestion and pipeline training.

### 1. Data Ingestion (Brief)

The `data_ingestion` module is responsible for downloading and processing the raw data. It has its own command-line interface (CLI) to start the process. For more details, refer to the `README.md` inside the `data_ingestion` directory.

To run the ingestion, navigate to the directory and execute the main script:
```bash
python data_ingestion/main.py <CLI_COMMANDS>
```

### 2. Training Pipeline

After the ingestion step is complete, the data will be ready to be processed and used for training the models. The `main.py` script in the project root orchestrates all steps of the main pipeline.

To run the full pipeline (processing, feature selection, and training), execute:
```bash
python main.py
```

## ⚙️ Configuration

Project settings, such as file paths, model parameters, and environment configurations, can be found and modified in the following locations:

- **Data Ingestion:** `data_ingestion/config/`
- **Main Pipeline:** `src/config/`