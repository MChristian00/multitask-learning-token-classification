# Enhancing Code-Switched Part-of-Speech Tagging and Language Identification using mBERT, XLM-R, and RemBERT with Uncertainty-Based Loss Weighting on LinCE CALCS

## Overview

This project investigates advanced fine-tuning techniques for multilingual transformer models—specifically **mBERT**, **XLM-RoBERTa (XLM-R)**, and **RemBERT**—to enhance Part-of-Speech (POS) tagging and Language Identification (LID) in English-Spanish code-switched (CS) text. The primary dataset is derived from the **LinCE CALCS English-Spanish benchmark**.

A key focus is the implementation and evaluation of **uncertainty-based multitask loss weighting** (Kendall et al., 2018). We compare this approach against standard fine-tuning paradigms:
* Single-Task Learning (ST)
* Joint Multitask Learning (JMT - Unweighted)
* Sequential Learning (LID-first then POS, and POS-first then LID)
* Joint Multitask Learning with static, manually-tuned loss weights (JMT - Static Weighted)

The objective is to determine if dynamically weighting task losses based on their learned homoscedastic uncertainty offers a more robust and effective method for training models on complex code-switched data.

## Motivation

Code-switched text presents unique challenges for NLP systems due to rapid language alternations and grammatical interplay. Accurate POS and LID are foundational for many downstream applications, and improving their performance on CS text is vital for linguistic equity and inclusivity for bilingual and multilingual communities. This project aims to contribute to more effective and equitable NLP tools by exploring tailored fine-tuning strategies.

## Setup 🔧

1.  **Install Python:** Ensure you have Python 3.9 or higher installed.
2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
3.  **Install Libraries:** Install the necessary libraries using pip. It's recommended to use a virtual environment.
    ```bash
    pip install torch transformers datasets evaluate seqeval numpy accelerate
    ```
4.  **GPU (Recommended):** A GPU (NVIDIA CUDA enabled) is highly recommended for reasonable training times.

## Data

* **Source Dataset**: The data for this project is based on the **LinCE (Linguistic Code-switching Evaluation) CALCS English-Spanish benchmark** (Aguilar et al., 2020). You will need to obtain this dataset from its official sources and preprocess it into the required format.
* **Local Structure**: Training, development, and test data files should be placed in the `data/` directory following the structure outlined in "Project Structure." For instance:
    * `data/POS/train.conll`, `data/POS/dev.conll`, `data/POS/test.conll`
    * `data/LID/train.conll`, `data/LID/dev.conll`, `data/LID/test.conll`
* **Format**: The scripts expect a CoNLL-like format. Each line should contain a token followed by its tag(s). For joint POS and LID data, a common format would be `token LID_tag POS_tag` per line, with sentences separated by blank lines. The `train_single.py` script might expect separate files for POS and LID tasks if labels are not combined.
* **Annotation Details (from LinCE CALCS En-Es)**:
    * **POS Tags**: Universal POS tagset (e.g., `ADJ`, `NOUN`, `VERB`, `PUNCT`, `X`, `UNK`).
    * **LID Labels**: CALCS scheme (e.g., `lang1` for English, `lang2` for Spanish, `mixed`, `ambiguous`, `fw`, `ne`, `unk`, `other`).
* **Preprocessing Applied in this Project's Experiments**:
    1.  Removal of non-speech artifacts.
    2.  Lowercasing of all tokens.
    3.  Filtering of short utterances.
    (These steps would be implemented in `src/utils/data_utils.py` or similar.)

## Configuration (`src/config/config.py`) ⚙️

The `src/config/config.py` file centralizes shared configuration settings for all training scripts. This includes:
* Model identifiers (e.g., `MODEL_NAME = "xlm-roberta-base"`)
* Dataset paths
* Training arguments (e.g., `LEARNING_RATE`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `NUM_TRAIN_EPOCHS`, `WEIGHT_DECAY`, `MAX_SEQUENCE_LENGTH`)
* Task-specific weights for static weighted multitask learning (e.g., `LID_WEIGHT`, `POS_WEIGHT`)
* A flag for enabling uncertainty-based loss weighting (e.g., `UNCERTAINTY_WEIGHTING = True/False`)

Ensure this file exists and is populated with your desired configurations before running training scripts.

## Models Used & Methodology

### Models
* **mBERT**: `bert-base-multilingual-cased` (Baseline)
* **XLM-RoBERTa (XLM-R)**: `xlm-roberta-base`
* **RemBERT**: `google/rembert`
    Each model uses its pretrained encoder with separate linear heads for POS and LID token classification.

### Fine-Tuning Strategies
1.  **Single-Task (ST)**: Separate models for POS and LID.
2.  **Sequential Learning (SEQ)**:
    * LID-first, then POS (SEQ L->P)
    * POS-first, then LID (SEQ P->L)
3.  **Joint Multitask Learning (JMT)**:
    * **Unweighted**: Sum of POS and LID losses.
    * **Static Weighted**: Pre-defined weights for POS and LID losses (set in `config.py`).
    * **Uncertainty-Based Weighted (Kendall et al., 2018)**: Dynamically weights task losses by learning task-specific homoscedastic uncertainty. The loss for task $i$ is $\mathcal{L}_i' = \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$, where $\sigma_i$ is a learnable noise parameter for task $i$. This is enabled by setting `UNCERTAINTY_WEIGHTING = True` in `config.py`.

## Running Experiments

Navigate to the project's base directory in your terminal.

* **Single-Task Training (POS and LID separately):**
    ```bash
    python src/train_single.py
    ```

* **Sequential-Task Training (POS-LID and LID-POS):**
    ```bash
    python src/train_sequential.py
    ```

* **Multitask Training (Unweighted, Static Weighted, or Uncertainty Weighted):**
    ```bash
    python src/train_multitask.py
    ```
    **Important Note for Multitask Training:**
    * For **uncertainty-based loss weighting**, ensure `UNCERTAINTY_WEIGHTING = True` in `src/config/config.py`.
    * For **unweighted multitasking**, ensure `UNCERTAINTY_WEIGHTING = False` and static weights (e.g., `LID_WEIGHT`, `POS_WEIGHT`) are set to `1.0` or effectively equal.
    * For **static-based loss weighting**, ensure `UNCERTAINTY_WEIGHTING = False` and modify `LID_WEIGHT` and `POS_WEIGHT` in `src/config/config.py` with the desired static values.

Training progress, logs, and model checkpoints will typically be saved to the `saves/` directory, as configured by Hugging Face `TrainingArguments` within the scripts (via `config.py`).

## Results and Analysis (Summary)

* **Model Comparison**: RemBERT and XLM-R are expected to outperform mBERT in single-task settings.
* **Single-Task vs. Multitask**: Single-task fine-tuning often provides strong baselines. Joint multitask learning (JMT) can sometimes suffer from task interference.
* **Sequential Learning**: May offer benefits by allowing the model to focus on one aspect of CS structure at a time (e.g., LID helping subsequent POS tagging).
* **Loss Weighting**:
    * Static weighting provides a way to manually balance tasks.
    * Uncertainty-based weighting is hypothesized to offer a more principled, data-driven approach to balancing tasks, potentially mitigating negative transfer in JMT and leading to more robust performance.

## Ethical Considerations

This research strives to enhance NLP capabilities for code-switching, promoting linguistic inclusivity. It's important to be mindful of potential biases in datasets and models, and the societal impact of language technologies.

## Future Work

* Integration of external linguistic features (e.g., morphological markers).
* Extension to other code-switched language pairs and domains (e.g., social media).
* Exploration of more advanced dynamic task weighting or curriculum learning strategies.