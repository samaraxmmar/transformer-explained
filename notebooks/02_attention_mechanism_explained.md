# Le mécanisme d'attention

Bienvenue dans le deuxième notebook de la série "Transformers Explained" ! Dans ce notebook, nous allons plonger au cœur de l'architecture des Transformers : le mécanisme d'attention, et plus particulièrement l'auto-attention (self-attention).

## Qu'est-ce que l'attention ?

Dans le contexte des réseaux de neurones, l'attention est un mécanisme qui permet au modèle de pondérer l'importance de différentes parties de la séquence d'entrée (ou de sortie) lors de la prédiction. Cela permet au modèle de se concentrer sur les informations les plus pertinentes, plutôt que de traiter toutes les informations de manière égale.

## L'auto-attention (Self-Attention)

L'auto-attention est un type d'attention où le modèle apprend à pondérer les relations entre les différentes positions d'une *même* séquence. Cela signifie que chaque mot dans une phrase peut "regarder" les autres mots de la phrase pour mieux comprendre son propre contexte.

### Comment ça marche ?

Le mécanisme d'auto-attention calcule trois vecteurs pour chaque mot d'entrée :

1.  **Query (Requête - Q)** : Représente ce que le mot actuel recherche.
2.  **Key (Clé - K)** : Représente ce que le mot actuel offre.
3.  **Value (Valeur - V)** : Contient l'information réelle du mot.

Le calcul de l'auto-attention se déroule en plusieurs étapes :

1.  **Calcul des scores de similarité** : Pour chaque mot, on calcule un score de similarité entre sa Query et les Keys de tous les autres mots (y compris lui-même). Cela se fait généralement par un produit scalaire.
2.  **Mise à l'échelle** : Les scores sont divisés par la racine carrée de la dimension des Keys (pour stabiliser les gradients).
3.  **Softmax** : Une fonction softmax est appliquée aux scores mis à l'échelle pour obtenir des poids d'attention. Ces poids indiquent l'importance de chaque mot pour le mot actuel.
4.  **Pondération des valeurs** : Les poids d'attention sont multipliés par les vecteurs Value correspondants. La somme de ces produits pondérés donne le vecteur de sortie pour le mot actuel.

### Formule de l'auto-attention

La formule de l'auto-attention est la suivante :

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Où :
-   $Q$ est la matrice des requêtes.
-   $K$ est la matrice des clés.
-   $V$ est la matrice des valeurs.
-   $d_k$ est la dimension des clés.

## Auto-attention multi-têtes (Multi-Head Attention)

L'auto-attention multi-têtes est une extension du mécanisme d'auto-attention. Au lieu d'effectuer un seul calcul d'attention, le modèle effectue plusieurs calculs d'attention en parallèle (chaque "tête" d'attention). Les résultats de ces têtes sont ensuite concaténés et transformés linéairement.

Cela permet au modèle de se concentrer sur différentes relations et aspects de la séquence simultanément, enrichissant ainsi sa capacité à capturer des dépendances complexes.

## Exemple Simplifié (Pseudo-code)

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query * key.T
        # sum over the last two dimensions
        energy = torch.einsum("nqhd,nkhd->nkhq", [queries, keys]) # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nkhq,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Dimensions du modèle
embed_size = 256
heads = 8
sequence_length = 10
batch_size = 2

# Création d'une instance de SelfAttention
attention_layer = SelfAttention(embed_size, heads)

# Exemple d'entrée (batch_size, sequence_length, embed_size)
# Supposons une séquence de 10 mots, avec un batch de 2
input_data = torch.rand(batch_size, sequence_length, embed_size)

# Passage à travers la couche d'attention
# Pour l'auto-attention, values, keys et query sont les mêmes
output = attention_layer(input_data, input_data, input_data, mask=None)

print(f"Shape de l'entrée: {input_data.shape}")
print(f"Shape de la sortie de l'attention: {output.shape}")
```

## Conclusion

Le mécanisme d'attention est la pierre angulaire des Transformers, leur permettant de traiter les informations de manière non séquentielle et de capturer des dépendances complexes. L'auto-attention multi-têtes améliore encore cette capacité en permettant au modèle de se concentrer sur diverses relations simultanément.

Dans le prochain notebook, nous utiliserons ces concepts pour construire un Transformer complet à partir de zéro.

