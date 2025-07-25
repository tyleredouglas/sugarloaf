from transformers import AutoTokenizer, AutoModel 
import numpy as np
import torch
import logging

class ChemBERTaEncoder:

    def __init__(self, 
                 model: str ='DeepChem/ChemBERTa-77M-MTR', 
                 device: str ='cpu'):
        """
        Initialize ChemBERTa encoder for molecular embeddings.
        Args:
            model: HuggingFace model name for ChemBERTa
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

    def _encode_chem(self, smiles: str) -> np.ndarray:
        """
        Retrieve ChemBERTa embedding for single molecule
        Args:
            smiles: molecule smiles
        """
        try:
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state 
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
                summed = torch.sum(hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts
        
            return mean_pooled.squeeze().numpy()
        
        except Exception as e:
            self.logger.warning(f"Failed to encode SMILES '{smiles}': {e}")
            return None  
        
    def encode_chem_batch(self, drug_dict: dict) -> dict[str, np.ndarray]: 
        """
        Retrieve ChemBERTa embedding for batch molecule
        Args:
            drug_dict: dictionary of drugs/cid/smiles from get_smiles()
        """
        chem_embeddings = {}
        failed_count = 0
        
        for drug, modality in drug_dict.items():
            smiles = modality['smiles']
            embedding = self._encode_chem(smiles)

            if embedding is not None:
                chem_embeddings[drug] = embedding
            else:
                failed_count += 1

        self.logger.info(f"Compounds encoded {len(chem_embeddings)}; Failed embeddings: {failed_count}")
        return chem_embeddings


