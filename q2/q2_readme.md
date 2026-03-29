## Steps to Reproduce the Experiments
All .PY files attached in zip file.

To reproduce the experiments, the following steps can be executed in a Google Colab environment.
Upload all .py file in google colab.


**Step 1: Create Directories and Install Dependencies**
```bash
!mkdir configs
!pip install torch torchaudio datasets librosa scikit-learn
```
This step prepares the working environment and installs all necessary libraries required for training and evaluation.

**Step 2: Train the Initial Model and Run Evaluation**
```bash
!python train.py
!python eval.py
```
This step performs the following tasks:
1. Trains the baseline speaker recognition model using `train.py`
2. Evaluates the trained model using `eval.py`
3. Saves the evaluation results

**Step 3: Train the Updated Model and Run Evaluation**
```bash
!python train_updated.py
!python eval.py
```
This step trains the improved version of the model using `train_updated.py`.
The evaluation script automatically loads the most recently saved model checkpoint for testing.

## Experimental Results
The results obtained from the evaluation process are summarized below.

**Model Performance Comparison**

| Model Training Script | Classification Accuracy | Equal Error Rate (EER) |
|-----------------------|-------------------------|------------------------|
| Model 1 (`train.py`)  | 0.9950                  | 0.0747                 |
| Model 2 (`train_updated.py`) | 0.9950                  | 0.0638                 |

**Result Analysis**
From the results above, both models achieve the same classification accuracy (0.9950). However, the key improvement can be observed in the Equal Error Rate (EER).
* Model 1 (Baseline) achieved an EER of 0.0747
* Model 2 (Improved Model) achieved a lower EER of 0.0638

A lower EER indicates better discrimination between genuine and impostor speaker attempts, which is an important metric in speaker verification systems.

Therefore, although the classification accuracy remains unchanged, Model 2 demonstrates improved verification performance, aligning with the objective of enhancing speaker recognition robustness.

**Checkpoint Results**
The results correspond to the checkpoints generated during the training process.

| Model Checkpoint | Source           | Accuracy | EER    |
|------------------|------------------|----------|--------|
| Model 1          | `train.py`       | 0.9950   | 0.0747 |
| Model 2          | `train_updated.py` | 0.9950   | 0.0638 |

The reduction in Equal Error Rate confirms that the updated training approach improves the model’s ability to distinguish between speakers while maintaining high accuracy.
