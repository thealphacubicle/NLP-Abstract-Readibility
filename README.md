# Readability Prediction Using NLP Models

This repository contains the code and resources for predicting the readability of research paper abstracts using three different models: Linear Regression, Gradient Boosted Random Forest Regressor, and fine-tuned DistilBERT. The models were evaluated on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values.

## Introduction

Predicting the readability of text is a challenging task that involves understanding the complexity and structure of language. This project aims to evaluate different models for this task and determine the most effective approach.

## Setup

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/readability-prediction.git
pip install -r requirements.txt
```

## Model Download

To load the model, follow these steps:

1. **Install the necessary libraries:**

    ```bash
    pip install transformers tensorflow
    ```

2. **Download the model weights:**

    Download the pre-trained model weights from [Hugging Face Model Hub](https://huggingface.co/models).

    ```python
    from transformers import TFDistilBertModel, DistilBertTokenizer

    model_name = "distilbert-base-uncased"
    model = TFDistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    ```

3. **Load the model and tokenizer using the provided metadata:**

    ```python
    import json

    # Load metadata
    with open('model/metadata.json', 'r') as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    model_type = metadata["model_type"]

    # Load tokenizer and model configuration
    tokenizer = DistilBertTokenizer.from_pretrained('model/tokenizer')
    model_config = model_type.from_pretrained('model/model_config')

    # Load model weights
    model = model_type.from_pretrained(model_name, config=model_config)
    ```

## Results

The models were evaluated based on their performance on the test dataset. The results are summarized as follows:

| Model                  | MSE           | MAE          | R-squared      |
|------------------------|---------------|--------------|----------------|
| DistilBERT Regressor   | 307.2758      | 13.497813    | -0.106518      |
| Linear Regression      | 4,906,583.0   | 1,868.485087 | -17,667.892255 |
| Random Forest          | 315.1672      | 13.684059    | -0.134936      |

## Acknowledgments

We would like to acknowledge the following resources and tools that made this project possible:

- Hugging Face. “Transformers.” *Hugging Face*, Hugging Face Inc., 2023, [https://huggingface.co/transformers/](https://huggingface.co/transformers/).
- TensorFlow. “TensorFlow Documentation.” *TensorFlow*, Google, 2023, [https://www.tensorflow.org/](https://www.tensorflow.org/).
- ArXiv. “ArXiv API.” *ArXiv*, Cornell University, 2023, [https://arxiv.org/help/api](https://arxiv.org/help/api).
- OpenAI. “ChatGPT.” *OpenAI*, OpenAI LP, 2023, [https://openai.com/chatgpt](https://openai.com/chatgpt).