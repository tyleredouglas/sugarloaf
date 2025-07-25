import scanpy as sc
import pandas as pd
import anndata as ad
import pyarrow.dataset as ds
import gcsfs
import requests
from io import StringIO
import re
import time
import urllib.parse
import logging

class DrugModalities:
    def __init__(self, 
                 query_rate=0.1, 
                 max_retries=3, 
                 cache_dir='data', 
                 fs = gcsfs.GCSFileSystem(), 
                 base_path='gs://arc-ctc-tahoe100/2025-02-25/'):
        
        """
        initialize DrugModalities for retrieving drug information from PubChem.
       
        args:
            query_rate: Sleep time between API requests
            max_retries: Maximum number of retry attempts for failed requests
            cache_dir: Directory for caching results
            fs: Google Cloud Storage filesystem object
            base_path: Base path to Tahoe 100M dataset in GCS (or other dataset)
        """

        self.query_rate = query_rate
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.base_path = base_path
        self.fs = fs
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _get_names(self) -> set[str]:

        """
        Returns set of unique drug names from Tahoe 100M metadata, configurable to other datasets
        """

        metadata_path =  "/".join([self.base_path.rstrip('/'), 'metadata', 'obs_metadata.parquet'])
        drugs = set(ds.dataset(metadata_path, filesystem=self.fs, format="parquet").to_table(columns=['drug']).to_pandas()['drug'].unique())
        return drugs

    def _clean_names(self, drug: str) -> str:

        """
        replace greek letters and special characters before API query
        
        args:
            drug: raw drug name string
            
        returns:
            cleaned drug name for PubChem API query
       """
        replacements = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ω': 'omega'}
        for symbol, spelled in replacements.items():
            if drug.startswith(symbol):
                drug = spelled + drug[1:]
            else:
                drug = drug.replace(symbol, f'-{spelled}')
        drug = drug.replace('/', ' ')

        return drug

    def _get_smiles_ind(self, drug: str) -> tuple[str, str]:
        
        """
        retrieve SMILES string and CID for a single drug from PubChem API.
            
        Args:
            drug: Clean drug name string
                
        Returns:
            Tuple of (smiles_string, cid) or error messages if retrieval fails
       """

        encoded_drug = urllib.parse.quote(drug)
        cid_url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_drug}/cids/TXT'
        smiles_url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_drug}/property/CanonicalSMILES/TXT'
        smiles = None
        cid = None

        smiles_response = requests.get(smiles_url)
        if smiles_response.status_code == 200:
            smiles = smiles_response.text.strip().splitlines()[0]
        elif smiles_response.status_code == 404:
            smiles = "not found"
        elif smiles_response.status_code == 503:
            for retry in range(1, self.max_retries + 1):
                time.sleep(self.query_rate * retry)
                retry_response = requests.get(smiles_url)
                if retry_response.status_code == 200:
                    smiles = retry_response.text.strip().splitlines()[0]
                    break
            else:
                smiles = "503 service unavailable"
        else:
            smiles = f'error: {smiles_response.status_code}'

    
        cid_response = requests.get(cid_url)
        if cid_response.status_code == 200:
            cid = cid_response.text.strip().splitlines()[0]
        elif cid_response.status_code == 404:
            cid = "not found"
        elif cid_response.status_code == 503:
            for retry in range(1, self.max_retries + 1):
                time.sleep(self.query_rate * retry)
                retry_response = requests.get(cid_url)
                if retry_response.status_code == 200:
                    cid = retry_response.text.strip().splitlines()[0]
                    break
            else:
                cid = "503 service unavailable"
        else:
            cid = f'error: {cid_response.status_code}'

        return smiles, cid
    
    def get_smiles(self, drug_list: list[str]) -> dict[str, dict[str, str]]:
         
        """
        retrieve SMILES strings and CIDs for a list of drugs from PubChem.
       
        Args:
            drug_list: List of drug names to process
           
        Returns:
           Dictionary mapping drug names to {'smiles': str, 'cid': str}.
           Failed drugs are excluded and logged.
         """
        results = {}
        failed_retrievals = []
    
        for drug in drug_list:
            cleaned_drug = self._clean_names(drug)
            smiles, cid = self._get_smiles_ind(cleaned_drug)  
            time.sleep(self.query_rate) 
            if smiles in ["not found", "503 service unavailable"] or smiles.startswith("error:"):
                failed_retrievals.append(drug)
            else:
                results[drug] = {"smiles": smiles, "cid": cid}

        logging.info(f"Failed drugs: {failed_retrievals}")
        return results

    