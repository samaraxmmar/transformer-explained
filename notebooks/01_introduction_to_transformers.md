# Introduction aux Transformers

Bienvenue dans le premier notebook de la série "Transformers Explained" ! Dans ce notebook, nous allons explorer les bases de l'architecture des Transformers, un modèle qui a révolutionné le traitement du langage naturel (TLN) et d'autres domaines.

## Qu'est-ce qu'un Transformer ?

Le Transformer est une architecture de réseau neuronal introduite en 2017 par Vaswani et al. dans l'article "Attention Is All You Need". Contrairement aux modèles précédents comme les RNN (réseaux de neurones récurrents) et les CNN (réseaux de neurones convolutifs), le Transformer s'appuie entièrement sur des mécanismes d'attention, éliminant ainsi la nécessité de récurrence ou de convolutions.

## Pourquoi les Transformers sont-ils importants ?

Avant les Transformers, les modèles séquentiels comme les RNN (LSTM, GRU) étaient dominants pour les tâches de TLN. Cependant, ils souffraient de limitations, notamment la difficulté à traiter les dépendances à longue portée et la parallélisation limitée. Les Transformers ont résolu ces problèmes en introduisant le mécanisme d'auto-attention, permettant au modèle de traiter toutes les parties d'une séquence simultanément.

## Architecture Générale

Un Transformer typique se compose de deux parties principales :

1.  **Encodeur** : Traite la séquence d'entrée et produit une représentation contextuelle.
2.  **Décodeur** : Utilise la représentation de l'encodeur pour générer la séquence de sortie.

Chaque encodeur et décodeur est composé de plusieurs couches identiques empilées.

### Blocs d'Encodeur

Chaque bloc d'encodeur contient deux sous-couches :

-   **Mécanisme d'auto-attention multi-têtes** : Permet au modèle de pondérer différentes parties de la séquence d'entrée.
-   **Réseau de neurones feed-forward positionnel** : Applique une transformation linéaire à chaque position.

### Blocs de Décodeur

Chaque bloc de décodeur contient trois sous-couches :

-   **Mécanisme d'auto-attention masqué multi-têtes** : Similaire à l'encodeur, mais masque les positions futures pour éviter la triche.
-   **Mécanisme d'attention multi-têtes Encodeur-Décodeur** : Permet au décodeur de se concentrer sur des parties pertinentes de la sortie de l'encodeur.
-   **Réseau de neurones feed-forward positionnel**.

## Encodage Positionnel

Puisque les Transformers ne contiennent pas de récurrence ou de convolution, ils n'ont intrinsèquement aucune notion de l'ordre des mots dans la séquence. Pour remédier à cela, des "encodages positionnels" sont ajoutés aux embeddings d'entrée. Ces encodages fournissent des informations sur la position relative ou absolue des tokens dans la séquence.

## Exemple Simplifié (Pseudo-code)

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Un exemple très simplifié d'une couche d'encodeur
class SimpleEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super(SimpleEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Dimensions du modèle
d_model = 512  # Dimension des embeddings
num_heads = 8  # Nombre de têtes d'attention
dim_feedforward = 2048 # Dimension du réseau feed-forward
dropout = 0.1

# Création d'une couche d'encodeur simple
encoder_layer = SimpleEncoderLayer(d_model, num_heads, dim_feedforward, dropout)

# Création d'un encodage positionnel
pos_encoder = PositionalEncoding(d_model)

# Exemple d'entrée (batch_size, sequence_length, d_model)
# Supposons une séquence de 10 mots, avec un batch de 2
input_sequence = torch.rand(10, 2, d_model)

# Ajout de l'encodage positionnel
input_with_pos = pos_encoder(input_sequence)

# Passage à travers la couche d'encodeur
output = encoder_layer(input_with_pos)

print(f"Shape de l'entrée: {input_sequence.shape}")
print(f"Shape de la sortie de l'encodeur: {output.shape}")
```

## Conclusion

Ce notebook a fourni une introduction aux concepts fondamentaux des Transformers. Dans les notebooks suivants, nous plongerons plus en détail dans le mécanisme d'attention et construirons un Transformer complet étape par étape.

