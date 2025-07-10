# 🧠 Fine-Tuning LLMs: Beginner to Advanced

This repository demonstrates **two approaches to fine-tuning large language models (LLMs)** using Hugging Face tools and LoRA. It is designed for beginners and intermediate users who want to understand how to adapt LLMs like LLaMA-2 to their own use cases.

---

## 📂 Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `LLM_Fine_Tune_Light.ipynb` | A simple, lightweight version to get you started without deep configuration |
| `finetuning_personal_dataset.ipynb` | A more flexible and customizable approach for advanced use cases |

---

## 🔍 Key Differences

| Feature | `LLM_Fine_Tune_Light.ipynb` | `finetuning_personal_dataset.ipynb` |
|--------|------------------------------|--------------------------------------|
| Dataset source | Hugging Face public dataset (`financial-qa-10k`) | Custom dataset (manual or private) |
| Data loading | `load_dataset()` from Hugging Face Hub | `Dataset.from_dict()` or from local files |
| Preprocessing | Minimal EDA (length stats) | Full formatting with instructions, inputs, responses |
| Fine-tuning style | Lightweight & quick | Fully customizable with LoRA configs |
| Best for | Beginners | Advanced users who want control |

---

## 📘 What's Happening in `LLM_Fine_Tune_Light.ipynb`

This notebook walks you through:
1. Loading a pre-hosted dataset (`financial-qa-10k`) using Hugging Face
2. Performing light EDA (avg. question/answer length)
3. Formatting it for instruction-style training
4. Fine-tuning a quantized LLaMA-2 model using LoRA

The model is loaded in **4-bit precision** using `BitsAndBytesConfig`, and training is done using **LoRA (Low-Rank Adaptation)** to make it memory-efficient.

---

## 🧪 What the Advanced Notebook Does

In `finetuning_personal_dataset.ipynb`, the user manually:
- Creates or loads a **custom dataset**
- Structures it in an **instruction-tuning** format: `{"instruction": ..., "input": ..., "output": ...}`
- Tokenizes and prepares it for fine-tuning
- Applies LoRA with target modules (`q_proj`, `k_proj`, `v_proj`, etc.)

This gives more control over how the model learns and adapts to your use case.

---

## 📂 How to Use Your Own Dataset

If you want to fine-tune with your own data in **`LLM_Fine_Tune_Light.ipynb`**, you can replace the dataset loading section.

### ✅ Option 1: From a `.csv` file
**File format:**
```csv
question,answer
What is inflation?,Inflation is...
What is GDP?,GDP stands for...
```

```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("/content/my_data.csv")
dataset = Dataset.from_pandas(df)
```

---

### ✅ Option 2: From a `.txt` file
**File format:**
```
Q: What is inflation? A: Inflation is...
Q: What is GDP? A: GDP stands for...
```

```python
from datasets import Dataset

with open("/content/my_data.txt") as f:
    lines = f.readlines()

data = {"question": [], "answer": []}
for line in lines:
    if line.startswith("Q:"):
        parts = line.split("A:")
        data["question"].append(parts[0].replace("Q:", "").strip())
        data["answer"].append(parts[1].strip())

dataset = Dataset.from_dict(data)
```

---

### ✅ Option 3: In-code Python dictionary

```python
from datasets import Dataset

data = {
    "question": ["What is inflation?", "What is GDP?"],
    "answer": ["Inflation is...", "GDP stands for..."]
}
dataset = Dataset.from_dict(data)
```

Then replace this line in the notebook:

```python
dataset = load_dataset("virattt/financial-qa-10K")
```

with:

```python
dataset = Dataset.from_dict(data)
```

---

## 🧠 What Is LoRA?

LoRA (Low-Rank Adaptation) is a technique that allows fine-tuning massive LLMs by only training a few layers — making it memory-efficient and feasible on smaller GPUs (like Colab T4 or A100). In our notebooks:
- Only key transformer layers like `q_proj`, `k_proj`, `v_proj`, etc., are updated
- Rest of the model stays frozen
- Works well with quantized models (e.g., 4-bit using BitsAndBytes)

---

## 💡 Summary

| Concept | Beginner Notebook | Advanced Notebook |
|--------|-------------------|------------------|
| Data | Public, pre-formatted | Private or custom |
| Preprocessing | Light (EDA only) | Full formatting |
| LoRA | Simple config | Deep customization |
| Purpose | Quick fine-tuning demo | Real-world customization |

---

## 🧩 Use Cases

You can adapt these notebooks for:
- Fine-tuning with **customer service chats** for retail
- Domain-specific Q&A like **medical or legal advice**
- Personalized assistants for **students, HR, or finance**
- Instruction-tuned bots for **internal enterprise tools**
