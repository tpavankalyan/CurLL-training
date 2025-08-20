# CurLL-v0: A Framework for Training and Evaluating Causal Language Models

This repository provides a comprehensive framework for training, evaluating, and analyzing causal language models (LLMs). The workflow is designed to be modular, allowing users to easily train models, run inference, evaluate performance, and collect results.

## Table of Contents

- [Project Overview](#project-overview)
- [Workflow](#workflow)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Training](#1-training)
  - [2. Uploading to Hugging Face Hub](#2-uploading-to-hugging-face-hub)
  - [3. Batched Inference](#3-batched-inference)
  - [4. Preparing Seed Data for Evaluation](#4-preparing-seed-data-for-evaluation)
  - [5. Evaluating the Model](#5-evaluating-the-model)
  - [6. Collecting Results](#6-collecting-results)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Pre-trained Models](#pre-trained-models)

## Project Overview

This project provides a set of scripts to automate the process of training and evaluating LLMs. The key features include:

-   **Training:** A flexible training script based on Hugging Face's `transformers` library.
-   **Inference:** A script for running batched inference with a trained model.
-   **Evaluation:** A script for evaluating the model's outputs using a separate LLM.
-   **Results Collection:** A script for collecting and organizing the evaluation results in a Google Sheet.

## Workflow

The workflow is divided into the following steps:

1.  **Training:** Train a causal language model using `run_train.py`.
2.  **Uploading to Hugging Face Hub:** Upload the trained model to the Hugging Face Hub using `upload_model_to_hf.py`.
3.  **Batched Inference:** Run batched inference with the trained model using `batched_inference.py`.
4.  **Preparing Seed Data for Evaluation:** Prepare the seed data for evaluation using `seed_for_eval.py`.
5.  **Evaluating the Model:** Evaluate the model's outputs using `llm_rating.py`.
6.  **Collecting Results:** Collect the evaluation results and upload them to a Google Sheet using `collect_results.py`.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/tpavankalyan/CurLL-v0.git
    cd CurLL-v0
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Google Sheets API credentials:**

    The `collect_results.py` script requires Google Sheets API credentials to upload the evaluation results. Follow these steps to set up the credentials:

    1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2.  Create a new project.
    3.  Enable the Google Sheets API and the Google Drive API.
    4.  Create a service account and download the JSON key file.
    5.  Rename the JSON key file to `agentdalal-6ea56a9e5ecc.json` and place it in the root directory of the project.

    **Note:** Be careful not to expose your service account credentials. Add the JSON key file to your `.gitignore` file to prevent it from being committed to version control.

## Usage

### 1. Training

The `run_train.py` script is used to train a causal language model. It takes a single command-line argument:

-   `--config`: The path to the configuration file (default: `config.yaml`).

**Example:**

```bash
python run_train.py --config configs/my_config.yaml
```

### 2. Uploading to Hugging Face Hub

The `upload_model_to_hf.py` script is used to upload a trained model to the Hugging Face Hub. It takes a single command-line argument:

-   `output_dir`: The path to the saved model directory.

**Example:**

```bash
python upload_model_to_hf.py /path/to/my_model
```

### 3. Batched Inference

The `batched_inference.py` script is used to run batched inference with a trained model. It takes the following command-line arguments:

-   `--model_path`: The path to the trained model.
-   `--data_type`: The type of data to use for inference (`instruct`, `cqa`, or `csqa`).
-   `--split_type`: The dataset split to use (`val` or `test`).
-   `--stage`: The stage number for the results.

**Example:**

```bash
python batched_inference.py --model_path Pavankalyan/my-model --data_type instruct --split_type val --stage 0
```

### 4. Preparing Seed Data for Evaluation

The `seed_for_eval.py` script is used to prepare the seed data for evaluation. It takes the following command-line arguments:

-   `--model_path`: The path to the trained model.
-   `--data_type`: The type of data to use (`instruct`, `cqa`, or `csqa`).
-   `--split_type`: The dataset split to use (`val` or `test`).
-   `--stage`: The stage number for the results.
-   `--root_dir`: The root directory for saving the seeds.

**Example:**

```bash
python seed_for_eval.py --model_path Pavankalyan/my-model --data_type instruct --split_type val --stage 0 --root_dir /path/to/seeds
```

### 5. Evaluating the Model

The `llm_rating.py` script is used to evaluate the model's outputs. It takes the following command-line arguments:

-   `--model_path`: The path to the trained model.
-   `--data_type`: The type of data to use (`instruct`, `cqa`, or `csqa`).
-   `--split_type`: The dataset split to use (`val` or `test`).
-   `--stage`: The stage number for the results.

**Example:**

```bash
python llm_rating.py --model_path Pavankalyan/my-model --data_type instruct --split_type val --stage 0
```

### 6. Collecting Results

The `collect_results.py` script is used to collect the evaluation results and upload them to a Google Sheet. It takes the following command-line arguments:

-   `--model_path`: The path to the trained model.
-   `--data_type`: The type of data to use (`instruct`, `cqa`, or `csqa`).
-   `--split_type`: The dataset split to use (`val` or `test`).
-   `--stage`: The stage number for the results.

**Example:**

```bash
python collect_results.py --model_path Pavankalyan/my-model --data_type instruct --split_type val --stage 0
```

## Configuration

The `run_train.py` script uses a YAML configuration file to specify the training parameters. See the `config.yaml` file for an example.

## Dependencies

The required dependencies are listed in the `requirements.txt` file.

## Pre-trained Models

The following pre-trained models are available on the Hugging Face Hub:

### Independent Models

-   [Pavankalyan/stage0](https://huggingface.co/Pavankalyan/stage0)
-   [Pavankalyan/stage1](https://huggingface.co/Pavankalyan/stage1)
-   [Pavankalyan/stage2](https://huggingface.co/Pavankalyan/stage2)
-   [Pavankalyan/stage3](https://huggingface.co/Pavankalyan/stage3)
-   [Pavankalyan/stage4](https://huggingface.co/Pavankalyan/stage4)

### Joint Models

-   [Pavankalyan/stage01](https://huggingface.co/Pavankalyan/stage01)
-   [Pavankalyan/stage012](https://huggingface.co/Pavankalyan/stage012)
-   [Pavankalyan/stage0123](https://huggingface.co/Pavankalyan/stage0123)
-   [Pavankalyan/stage01234](https://huggingface.co/Pavankalyan/stage01234)

### Continual Models

-   [Pavankalyan/stage0_1](https://huggingface.co/Pavankalyan/stage0_1)
-   [Pavankalyan/stage0_1_2](https://huggingface.co/Pavankalyan/stage0_1_2)
-   [Pavankalyan/stage0_1_2_3](https://huggingface.co/Pavankalyan/stage0_1_2_3)
-   [Pavankalyan/stage0_1_2_3_4](https://huggingface.co/Pavankalyan/stage0_1_2_3_4)
