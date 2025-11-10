import pandas as pd
from pronto import Ontology
from Bio import SeqIO
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Data Processing

def prepare_seq(seq):
    spaced_seq = " ".join(list(seq.upper()))
    formatted_seq = f"[CLS] {spaced_seq} [SEP]"
    return formatted_seq

def get_protbert_embeddings(sequences, model, tokenizer, device):
    ids = tokenizer(sequences, add_special_tokens=False, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model(**ids)

    embeddings = output.last_hidden_state

    protein_embeddings = embeddings[:, 0, :]

    return protein_embeddings.cpu().numpy()

def fasta_parse(fasta_file):
    records = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        records.append({
            "ID": seq_record.id,
            "Sequence": prepare_seq(str(seq_record.seq))
        })

    return pd.DataFrame(records)

# Training Sets
taxonomy_df = pd.read_csv('cafa-6-protein-function-prediction copy/Train/train_taxonomy.tsv', sep='\t')

terms_df = pd.read_csv('cafa-6-protein-function-prediction copy/Train/train_terms.tsv', sep='\t')

go_basic = Ontology('cafa-6-protein-function-prediction copy/Train/go-basic.obo')

fasta_file = 'cafa-6-protein-function-prediction copy/Train/train_sequences.fasta'
seq_df = fasta_parse(fasta_file)

print(seq_df.head())
print(taxonomy_df)
print(terms_df)
print(f"Number of terms: {len(go_basic.terms())}")

# Testing Sets
superset_taxon_df = pd.read_csv('cafa-6-protein-function-prediction copy/Test/testsuperset-taxon-list.tsv', sep='\t')

fasta_file = 'cafa-6-protein-function-prediction copy/Test/testsuperset.fasta'
superset_seq_df = fasta_parse(fasta_file)


# --- 2a: Filter GO Terms (Crucial Step) ---
# The ontology has ~40k terms; you need to pick a subset of frequent ones to predict.
# Let's start with the top N most frequent terms.
top_n = 2000
term_counts = terms_df['term'].value_counts()
selected_terms = term_counts.head(top_n).index.tolist()

# --- 2b: Group terms by protein accession ---
# Consolidate all terms for each protein ID into a single list
protein_terms = terms_df.groupby('Protein Accession')['term'].apply(list).reset_index()

# --- 2c: Filter the terms list to only include our selected terms ---
protein_terms['filtered_terms'] = protein_terms['term'].apply(
    lambda x: [term for term in x if term in selected_terms]
)
# Remove proteins that have none of the selected terms after filtering
protein_terms = protein_terms[protein_terms['filtered_terms'].map(len) > 0]


# --- 2d: Create the MultiLabelBinarizer (MLiB) and the binary matrix ---
# The MLiB maps each GO term string to a column index
mlb = MultiLabelBinarizer(classes=selected_terms)
y_labels = mlb.fit_transform(protein_terms['filtered_terms'])

print(f"Shape of Y labels matrix: {y_labels.shape}")



# Model prep
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertModel.from_pretrained("Rostlab/prot_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode
sequences = seq_df['Sequence'].tolist()

protein_features = get_protbert_embeddings(sequences, model, tokenizer, device)
print(f"Shape of features: {protein_features.shape}")

# The protein IDs corresponding to the Y matrix
y_protein_ids = protein_terms['Protein Accession'].tolist()

# Ensure seq_df is aligned and filtered to match the proteins we have labels for
seq_df_aligned = seq_df[seq_df['ID'].isin(y_protein_ids)].copy()
# Reorder the sequence dataframe rows to match the order in y_protein_ids
seq_df_aligned['ID'] = pd.Categorical(seq_df_aligned['ID'], categories=y_protein_ids, ordered=True)
seq_df_aligned = seq_df_aligned.sort_values('ID').reset_index(drop=True)

# Re-run embedding generation *only* on the aligned sequences if necessary,
# or ensure the original protein_features array matches this order exactly.
# The simplest approach is to use the aligned sequence list:

aligned_sequences = seq_df_aligned['Sequence'].tolist()
# This might take a moment to rerun:
X_features = get_protbert_embeddings(aligned_sequences, model, tokenizer, device)


print(f"Shape of X features: {X_features.shape}")
print(f"Shape of Y labels: {y_labels.shape}")

# Split data (using the aligned X and Y)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Define Base Learners (e.g., Random Forest/Bagging, XGBoost/Boosting)
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)

estimators = [
    ('rf', OneVsRestClassifier(rf)),
    ('xgb', OneVsRestClassifier(xgb))
]

# Define Meta Learner
meta_learner = LogisticRegression(max_iter=1000)

# Create the Stacking Classifier with OneVsRest for multi-label compatibility
# Note: StackingClassifier natively supports multi-label, so we remove the OVR from the base
stacked_ensemble = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)], # Use raw models here
    final_estimator=OneVsRestClassifier(meta_learner), # Wrap meta-learner in OVR
    cv=3,
    n_jobs=-1,
    passthrough=False
)

print("\nStarting ensemble training...")
# Train the model
stacked_ensemble.fit(X_train, y_train)
print("Training complete.")