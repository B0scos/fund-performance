# Data Clustering Project

This project is a complete solution for data clustering, from initial data ingestion and processing to the training and evaluation of machine learning models. The system is divided into two main components: a data ingestion module and a clustering model training pipeline.

## ✨ Features

- **Data Ingestion Module:** Automatically collects and prepares raw data.
- **Processing and Cleaning:** Pipelines to validate, clean, and transform data.
- **Feature Engineering:** Creation and selection of features to optimize model performance.
- **Model Training:** Support for multiple clustering algorithms, such as K-Means and Gaussian Mixture Models (GMM).
- **Modular Structure:** Code organized into reusable components, facilitating maintenance and expansion.

## 📂 Project Structure

The project is organized into the following main directories:

- **`/data_ingestion`**: Module responsible for the initial collection and storage of data. It contains its own logic, CLI, and configurations.
- **`/src`**: Contains the main application code, including processing pipelines, model training, and utilities.
- **`/notebooks`**: Jupyter Notebooks for exploratory analysis, testing, and prototyping.
- **`/main.py`**: Main entry point to orchestrate the project's pipelines.
- **`/requirements.txt`**: List of the project's Python dependencies.

## 🚀 Getting Started

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