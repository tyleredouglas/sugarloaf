from  src.data.smiles import DrugModalities
from  src.embeddings.chemberta import ChemBERTaEncoder

drug = ['Sinomenine']

retriever = DrugModalities()
encoder = ChemBERTaEncoder()

modalities = retriever.get_smiles(drug)
embeddings = encoder.encode_chem_batch(modalities)

print(f"Retrieved SMILES for {len(modalities)} drugs")
for drug, modality in modalities.items():
    print(modality['smiles'])

print(f"Generated embeddings for {len(embeddings)} drugs")
for drug, embedding in embeddings.items():
    print(drug, embedding)

