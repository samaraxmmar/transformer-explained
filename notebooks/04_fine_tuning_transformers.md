# Fine-tuning des Transformers

Bienvenue dans le quatrième et dernier notebook de la série "Transformers Explained" ! Dans ce notebook, nous allons explorer l'un des aspects les plus puissants des Transformers : le fine-tuning. Le fine-tuning permet d'adapter des modèles de Transformers pré-entraînés à des tâches spécifiques avec des ensembles de données plus petits, ce qui est extrêmement efficace et courant dans le domaine du TLN.

## Qu'est-ce que le Fine-tuning ?

Le fine-tuning est un processus qui consiste à prendre un modèle de réseau neuronal qui a déjà été entraîné sur un grand ensemble de données (souvent appelé pré-entraînement) et à l'entraîner davantage sur un ensemble de données plus petit et spécifique à une tâche. Pour les Transformers, cela signifie généralement prendre un modèle comme BERT, GPT, ou T5, qui a été pré-entraîné sur des téraoctets de texte, et l'adapter à des tâches comme la classification de texte, la reconnaissance d'entités nommées, la traduction, etc.

## Pourquoi le Fine-tuning est-il si efficace ?

1.  **Transfert de connaissances** : Les modèles pré-entraînés ont déjà appris des représentations riches et générales du langage à partir de vastes corpus de texte. Ces connaissances sont transférées à la nouvelle tâche.
2.  **Moins de données nécessaires** : Puisque le modèle a déjà une bonne compréhension du langage, il nécessite beaucoup moins de données spécifiques à la tâche pour atteindre de bonnes performances, par rapport à l'entraînement d'un modèle à partir de zéro.
3.  **Temps d'entraînement réduit** : Le fine-tuning est généralement beaucoup plus rapide que l'entraînement à partir de zéro, car le modèle a déjà convergé vers une bonne solution.

## Processus de Fine-tuning

Le processus général de fine-tuning implique les étapes suivantes :

1.  **Choisir un modèle pré-entraîné** : Sélectionner un modèle de Transformer (par exemple, `bert-base-uncased`, `gpt2`, `t5-small`) adapté à votre tâche et à vos ressources.
2.  **Préparer les données** : Adapter votre ensemble de données spécifique à la tâche au format attendu par le modèle (tokenisation, ajout de tokens spéciaux, etc.).
3.  **Modifier la couche de sortie** : Remplacer la couche de sortie du modèle pré-entraîné par une nouvelle couche adaptée à votre tâche (par exemple, une couche de classification pour la classification de texte).
4.  **Entraîner le modèle** : Entraîner le modèle sur votre ensemble de données spécifique à la tâche, en ajustant les poids de toutes les couches (ou seulement des dernières couches) avec un taux d'apprentissage faible.

## Exemple de Fine-tuning avec Hugging Face Transformers

La bibliothèque Hugging Face `transformers` est l'outil le plus populaire pour travailler avec les Transformers. Elle simplifie grandement le processus de fine-tuning.

Nous allons montrer un exemple simplifié de fine-tuning pour une tâche de classification de texte en utilisant un modèle BERT.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. Charger un modèle pré-entraîné et un tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # 2 classes: positif/négatif

# 2. Préparer les données
# Utilisons un petit dataset de classification de sentiments pour l'exemple
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_imdb = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_imdb["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_imdb["test"].shuffle(seed=42).select(range(200))

# Renommer la colonne 'label' en 'labels' pour être compatible avec Trainer
small_train_dataset = small_train_dataset.rename_column("label", "labels")
small_eval_dataset = small_eval_dataset.rename_column("label", "labels")

# Supprimer les colonnes inutiles
small_train_dataset = small_train_dataset.remove_columns(["text"])
small_eval_dataset = small_eval_dataset.remove_columns(["text"])

# Définir le format des données pour PyTorch
small_train_dataset.set_format("torch")
small_eval_dataset.set_format("torch")

# 3. Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 4. Définir une fonction de métrique (optionnel mais recommandé)
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Créer l'objet Trainer et entraîner le modèle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("Début du fine-tuning...")
trainer.train()
print("Fine-tuning terminé.")

# Évaluer le modèle
print("Évaluation du modèle...")
eval_results = trainer.evaluate()
print(f"Résultats de l'évaluation: {eval_results}")

# Exemple de prédiction
text_to_classify = "This movie was absolutely fantastic and I loved every minute of it!"
inputs = tokenizer(text_to_classify, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
predicted_label = "positif" if predicted_class_id == 1 else "négatif"

print(f"Texte: \"{text_to_classify}\"")
print(f"Sentiment prédit: {predicted_label}")

text_to_classify_neg = "This was a terrible film, I hated it."
inputs_neg = tokenizer(text_to_classify_neg, return_tensors="pt")

with torch.no_grad():
    logits_neg = model(**inputs_neg).logits

predicted_class_id_neg = logits_neg.argmax().item()
predicted_label_neg = "positif" if predicted_class_id_neg == 1 else "négatif"

print(f"Texte: \"{text_to_classify_neg}\"")
print(f"Sentiment prédit: {predicted_label_neg}")
```

## Conclusion

Le fine-tuning est une technique essentielle pour tirer parti de la puissance des modèles de Transformers pré-entraînés. Il permet d'obtenir des performances de pointe sur une grande variété de tâches de TLN avec des efforts et des ressources d'entraînement considérablement réduits.

Ce dépôt vous a fourni une base solide pour comprendre les Transformers, de leurs principes fondamentaux à leur application pratique via le fine-tuning. Nous espérons que cela vous aidera dans votre parcours d'apprentissage du TLN !

