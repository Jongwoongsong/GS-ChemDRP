import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class GDSCDatasetGS(Dataset):
    """
    GDSC Dataset for GS Model
    
    CSV 구조 (955 columns):
        - CELL_LINE_NAME
        - DRUG_NAME  
        - MIN_CONC, MAX_CONC
        - LN_IC50 (label)
        - canonical_smiles
        - 949 gene expression columns (Entrez IDs)
    """
    def __init__(
        self,
        csv_path: str,
        gene_cols: Optional[List[str]] = None,
        normalize_expr: bool = True,
        cache_data: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.normalize_expr = normalize_expr
        
        print(f"[Dataset] Loading {self.csv_path.name}...")
        self.df = pd.read_csv(csv_path)
        print(f"[Dataset] Loaded {len(self.df):,} samples")
        
        # Meta columns
        self.meta_cols = ['CELL_LINE_NAME', 'DRUG_NAME', 'MIN_CONC', 'MAX_CONC', 'LN_IC50', 'canonical_smiles']
        
        # Auto-detect gene columns
        if gene_cols is None:
            self.gene_cols = [c for c in self.df.columns if c not in self.meta_cols]
        else:
            self.gene_cols = gene_cols
        
        self.num_genes = len(self.gene_cols)
        print(f"[Dataset] {self.num_genes} genes detected")
        
        # Extract gene expression matrix
        self.gene_expr = self.df[self.gene_cols].values.astype(np.float32)
        
        # Normalize gene expression
        if normalize_expr:
            self.expr_mean = self.gene_expr.mean(axis=0)
            self.expr_std = self.gene_expr.std(axis=0) + 1e-8
            self.gene_expr = (self.gene_expr - self.expr_mean) / self.expr_std
            print(f"[Dataset] Gene expression normalized")
        
        # Extract other columns
        self.smiles = self.df['canonical_smiles'].tolist()
        self.labels = self.df['LN_IC50'].values.astype(np.float32)
        self.cell_names = self.df['CELL_LINE_NAME'].tolist()
        self.drug_names = self.df['DRUG_NAME'].tolist()
        
        # Cache as tensors
        if cache_data:
            self.gene_expr_tensor = torch.from_numpy(self.gene_expr)
            self.labels_tensor = torch.from_numpy(self.labels)
        else:
            self.gene_expr_tensor = None
            self.labels_tensor = None
        
        print(f"[Dataset] Ready! Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.gene_expr_tensor is not None:
            gene_expr = self.gene_expr_tensor[idx]
            label = self.labels_tensor[idx]
        else:
            gene_expr = torch.from_numpy(self.gene_expr[idx])
            label = torch.tensor(self.labels[idx])
        
        return {
            'smiles': self.smiles[idx],
            'gene_expr': gene_expr,
            'label': label,
            'cell_name': self.cell_names[idx],
            'drug_name': self.drug_names[idx],
        }
    
    def get_normalization_stats(self) -> Dict:
        return {
            'mean': self.expr_mean if self.normalize_expr else None,
            'std': self.expr_std if self.normalize_expr else None,
        }


def collate_fn_gs(batch: List[Dict]) -> Dict:
    """Custom collate function for GS Dataset"""
    smiles_list = [item['smiles'] for item in batch]
    gene_expr = torch.stack([item['gene_expr'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    cell_names = [item['cell_name'] for item in batch]
    drug_names = [item['drug_name'] for item in batch]
    
    return {
        'smiles_list': smiles_list,
        'gene_expr': gene_expr,
        'labels': labels,
        'cell_names': cell_names,
        'drug_names': drug_names,
    }


def create_dataloaders(
    train_path: str,
    valid_path: str,
    test_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    normalize_expr: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train/valid/test dataloaders"""
    
    train_dataset = GDSCDatasetGS(train_path, normalize_expr=normalize_expr)
    
    if normalize_expr:
        valid_dataset = GDSCDatasetGS(valid_path, normalize_expr=False)
        test_dataset = GDSCDatasetGS(test_path, normalize_expr=False)
        
        # Apply train normalization to valid/test
        train_stats = train_dataset.get_normalization_stats()
        valid_dataset.gene_expr = (valid_dataset.gene_expr - train_stats['mean']) / train_stats['std']
        test_dataset.gene_expr = (test_dataset.gene_expr - train_stats['mean']) / train_stats['std']
        
        valid_dataset.gene_expr_tensor = torch.from_numpy(valid_dataset.gene_expr)
        test_dataset.gene_expr_tensor = torch.from_numpy(test_dataset.gene_expr)
    else:
        valid_dataset = GDSCDatasetGS(valid_path, normalize_expr=False)
        test_dataset = GDSCDatasetGS(test_path, normalize_expr=False)
        train_stats = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_gs, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_gs, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_gs, pin_memory=True,
    )
    
    info = {
        'num_genes': train_dataset.num_genes,
        'gene_cols': train_dataset.gene_cols,
        'train_size': len(train_dataset),
        'valid_size': len(valid_dataset),
        'test_size': len(test_dataset),
        'norm_stats': train_stats,
    }
    
    return train_loader, valid_loader, test_loader, info