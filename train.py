import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import model
from gs_geneset_gating import GSGeneSetGatingModel, count_parameters


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DrugCellDataset(Dataset):
    """Dataset for drug-cell response prediction with pathway labels"""
    
    def __init__(
        self,
        csv_path: str,
        gene_columns: List[str],
        drug_pathway_map: Optional[Dict[str, str]] = None,
        smiles_col: str = "canonical_smiles",
        target_col: str = "LN_IC50",
        drug_name_col: str = "DRUG_NAME",  # Drug name column for pathway matching
    ):
        self.df = pd.read_csv(csv_path)
        self.gene_columns = gene_columns
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.drug_name_col = drug_name_col
        
        # Pathway mapping for contrastive learning (drug_name -> pathway)
        self.drug_pathway_map = drug_pathway_map
        self.pathway_to_idx = {}
        self.drug_pathway_idx = {}  # drug_name -> pathway_idx
        
        if drug_pathway_map:
            # Build pathway index
            pathways = sorted(set(drug_pathway_map.values()))
            self.pathway_to_idx = {p: i for i, p in enumerate(pathways)}
            
            # Map each drug name to pathway index
            for drug_name, pathway in drug_pathway_map.items():
                if pathway in self.pathway_to_idx:
                    self.drug_pathway_idx[drug_name] = self.pathway_to_idx[pathway]
            
            # Count samples with pathway info
            if drug_name_col in self.df.columns:
                valid_count = sum(1 for d in self.df[drug_name_col] if d in self.drug_pathway_idx)
                unique_drugs = len(set(self.df[drug_name_col]) & set(drug_pathway_map.keys()))
            else:
                valid_count = 0
                unique_drugs = 0
            
            print(f"[Dataset] {len(self.pathway_to_idx)} pathways, {unique_drugs} drugs ({valid_count} samples) with pathway info")
        
        print(f"[Dataset] Loaded {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        smiles = row[self.smiles_col]
        gene_expr = torch.tensor(row[self.gene_columns].values.astype(np.float32))
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        # Pathway label (-1 if not available) - match by drug name
        pathway_idx = -1
        if self.drug_name_col in self.df.columns:
            drug_name = row[self.drug_name_col]
            pathway_idx = self.drug_pathway_idx.get(drug_name, -1)
        
        return {
            "smiles": smiles,
            "gene_expr": gene_expr,
            "target": target,
            "pathway_idx": pathway_idx,
        }


def collate_fn(batch):
    return {
        "smiles": [x["smiles"] for x in batch],
        "gene_expr": torch.stack([x["gene_expr"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "pathway_idx": torch.tensor([x["pathway_idx"] for x in batch]),
    }


# ─── Contrastive Loss ─────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] normalized embeddings
            labels: [N] pathway labels (-1 for no label)
        """
        device = features.device
        
        # Filter out samples without pathway labels
        mask = labels >= 0
        if mask.sum() < 2:
            return torch.tensor(0.0, device=device)
        
        features = features[mask]
        labels = labels[mask]
        
        # Normalize
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [N, N]
        
        # Mask for positive pairs (same pathway)
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.T).float()  # [N, N]
        
        # Remove diagonal
        eye = torch.eye(pos_mask.size(0), device=device)
        pos_mask = pos_mask - eye
        
        # Log-softmax over all pairs except self
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()  # For numerical stability
        
        exp_logits = torch.exp(logits) * (1 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean over positive pairs
        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
        
        return loss.mean()


# ─── Training Functions ───────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gene_set_masks: torch.Tensor,
    scaler: Optional[GradScaler] = None,
    supcon_loss_fn: Optional[nn.Module] = None,
    lambda_con: float = 0.1,
    lambda_gate_l1: float = 0.01,
    lambda_gate_smooth: float = 0.01,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_mse = 0.0
    total_con = 0.0
    total_gate_l1 = 0.0
    total_gate_smooth = 0.0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        smiles = batch["smiles"]
        gene_expr = batch["gene_expr"].to(device)
        target = batch["target"].to(device)
        pathway_idx = batch["pathway_idx"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            out = model(smiles, gene_expr, gene_set_masks)
            pred = out["pred"]
            
            # MSE loss
            mse_loss = F.mse_loss(pred, target)
            
            # Contrastive loss
            con_loss = torch.tensor(0.0, device=device)
            if supcon_loss_fn is not None and out["drug_emb"] is not None:
                con_loss = supcon_loss_fn(out["drug_emb"], pathway_idx)
            
            # Gate regularization losses
            aux_losses = model.get_auxiliary_losses(
                out["gate_values"], 
                gene_set_masks,
                lambda_l1=lambda_gate_l1,
                lambda_smooth=lambda_gate_smooth,
            )
            gate_l1 = aux_losses.get("gate_l1", torch.tensor(0.0, device=device))
            gate_smooth = aux_losses.get("gate_smooth", torch.tensor(0.0, device=device))
            
            # Total loss
            loss = mse_loss + lambda_con * con_loss + gate_l1 + gate_smooth
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate
        total_mse += mse_loss.item() * len(smiles)
        total_con += con_loss.item() * len(smiles)
        total_gate_l1 += gate_l1.item() * len(smiles)
        total_gate_smooth += gate_smooth.item() * len(smiles)
        
        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # Progress bar
        pbar.set_postfix({
            "mse": f"{mse_loss.item():.4f}",
            "con": f"{con_loss.item():.4f}",
        })
    
    # Compute metrics
    n = len(all_preds)
    pcc = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return {
        "mse": total_mse / n,
        "con": total_con / n,
        "gate_l1": total_gate_l1 / n,
        "gate_smooth": total_gate_smooth / n,
        "pcc": pcc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    gene_set_masks: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    
    all_preds, all_targets = [], []
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        smiles = batch["smiles"]
        gene_expr = batch["gene_expr"].to(device)
        target = batch["target"].to(device)
        
        out = model(smiles, gene_expr, gene_set_masks)
        pred = out["pred"]
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    pcc = np.corrcoef(all_preds, all_targets)[0, 1]
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return {"pcc": pcc, "rmse": rmse}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GS with Gene Gating")
    
    # Data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--gene_set_path", type=str, required=True, help="JSON file with gene sets")
    parser.add_argument("--pathway_mapping", type=str, default=None, help="Drug pathway mapping JSON")
    parser.add_argument("--smiles_col", type=str, default="canonical_smiles")
    parser.add_argument("--target_col", type=str, default="LN_IC50")
    parser.add_argument("--drug_name_col", type=str, default="DRUG_NAME")
    
    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_gene_gating", action="store_true", default=True)
    parser.add_argument("--no_gene_gating", action="store_false", dest="use_gene_gating")
    parser.add_argument("--use_geneset_residual", action="store_true", default=True)
    parser.add_argument("--no_geneset_residual", action="store_false", dest="use_geneset_residual")
    parser.add_argument("--geneset_residual_init", type=float, default=0.1)
    
    # Loss weights
    parser.add_argument("--lambda_con", type=float, default=0.1)
    parser.add_argument("--lambda_gate_l1", type=float, default=0.01)
    parser.add_argument("--lambda_gate_smooth", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs_gs_gating")
    parser.add_argument("--run_name", type=str, default="gating_exp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {run_dir}")
    
    # ─── Load Gene Columns from CSV ──────────────────────────────────────────
    # Get gene columns from training data
    train_df = pd.read_csv(args.train_path, nrows=1)
    
    # Find gene columns (exclude metadata columns)
    exclude_cols = {args.smiles_col, args.target_col, args.drug_name_col,
                    'cell_line', 'drug_name', 'cell_iname', 'drug_id', 
                    'CELL_LINE_NAME', 'DRUG_NAME', 'DRUG_ID',
                    'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE',
                    'Unnamed: 0', 'index'}
    
    # Gene columns are numeric strings (Entrez IDs)
    gene_columns = [c for c in train_df.columns 
                    if c not in exclude_cols and c.replace('.', '').replace('-', '').isdigit()]
    
    if len(gene_columns) == 0:
        # Fallback: use all columns except known metadata
        gene_columns = [c for c in train_df.columns if c not in exclude_cols]
    
    num_genes = len(gene_columns)
    gene_to_idx = {g: i for i, g in enumerate(gene_columns)}
    
    print(f"[Data] {num_genes} genes")
    
    # ─── Load Gene Sets ───────────────────────────────────────────────────────
    with open(args.gene_set_path, "r") as f:
        gene_set_data = json.load(f)
    
    # Handle different JSON formats
    if "set_names" in gene_set_data:
        # Format: {"set_names": [...], "gene_to_idx": {...}, "set_to_genes": {...}}
        set_names = gene_set_data["set_names"]
        set_to_genes = gene_set_data["set_to_genes"]
    else:
        # Format: {"SET_NAME": [gene1, gene2, ...], ...}
        set_names = list(gene_set_data.keys())
        set_to_genes = gene_set_data
    
    num_sets = len(set_names)
    
    # Build gene set masks
    gene_set_masks = torch.zeros(num_sets, num_genes)
    genes_matched = 0
    for s, set_name in enumerate(set_names):
        genes_in_set = set_to_genes[set_name]
        for gene in genes_in_set:
            gene_str = str(gene)  # Ensure string
            if gene_str in gene_to_idx:
                gene_set_masks[s, gene_to_idx[gene_str]] = 1
                genes_matched += 1
    gene_set_masks = gene_set_masks.to(device)
    
    avg_genes = gene_set_masks.sum(dim=1).mean().item()
    total_genes_in_sets = sum(len(set_to_genes[s]) for s in set_names)
    print(f"[GeneSet] {num_sets} sets, avg genes/set: {avg_genes:.1f}")
    print(f"[GeneSet] Matched {genes_matched}/{total_genes_in_sets} gene-set mappings")
    
    # ─── Load Pathway Mapping ─────────────────────────────────────────────────
    drug_pathway_map = None
    if args.pathway_mapping:
        with open(args.pathway_mapping, "r") as f:
            raw_mapping = json.load(f)
        
        # Handle different JSON structures
        if "drug_to_pathway" in raw_mapping:
            # Format: {"drug_to_pathway": {"Erlotinib": "EGFR signaling", ...}}
            drug_pathway_map = raw_mapping["drug_to_pathway"]
        elif isinstance(raw_mapping, dict):
            # Check if values are strings (direct mapping) or dicts (nested)
            first_val = next(iter(raw_mapping.values()), None)
            if isinstance(first_val, str):
                # Format: {"Erlotinib": "EGFR signaling", ...}
                drug_pathway_map = raw_mapping
            elif isinstance(first_val, dict):
                # Format: {"drug_name": {"pathway": "...", ...}, ...}
                drug_pathway_map = {}
                for drug, info in raw_mapping.items():
                    pathway = info.get("TARGET_PATHWAY", info.get("pathway"))
                    if pathway:
                        drug_pathway_map[drug] = pathway
        
        if drug_pathway_map:
            print(f"[Contrastive] {len(drug_pathway_map)} drugs with pathway labels")
            # Show sample mappings
            sample_drugs = list(drug_pathway_map.items())[:3]
            for drug, pathway in sample_drugs:
                print(f"  - {drug}: {pathway}")
        else:
            print("[Contrastive] No valid pathway mapping found")
    
    # ─── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = DrugCellDataset(
        args.train_path, gene_columns, drug_pathway_map,
        args.smiles_col, args.target_col, args.drug_name_col
    )
    valid_dataset = DrugCellDataset(
        args.valid_path, gene_columns, drug_pathway_map,
        args.smiles_col, args.target_col, args.drug_name_col
    )
    test_dataset = DrugCellDataset(
        args.test_path, gene_columns, drug_pathway_map,
        args.smiles_col, args.target_col, args.drug_name_col
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # ─── Model ────────────────────────────────────────────────────────────────
    model = GSGeneSetGatingModel(
        num_genes=num_genes,
        num_gene_sets=num_sets,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_gene_gating=args.use_gene_gating,
        use_geneset_residual=args.use_geneset_residual,
        geneset_residual_init=args.geneset_residual_init,
    ).to(device)
    
    trainable = count_parameters(model)
    print(f"[Model] Trainable params: {trainable:,}")
    print(f"[Model] Gene Gating: {args.use_gene_gating}, GeneSet Residual: {args.use_geneset_residual}")
    
    # ─── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP
    scaler = GradScaler() if args.amp else None
    
    # Contrastive loss
    supcon_loss_fn = SupConLoss(temperature=args.temperature) if drug_pathway_map else None
    
    # ─── Training Loop ────────────────────────────────────────────────────────
    best_val_pcc = -1.0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, gene_set_masks,
            scaler=scaler,
            supcon_loss_fn=supcon_loss_fn,
            lambda_con=args.lambda_con,
            lambda_gate_l1=args.lambda_gate_l1,
            lambda_gate_smooth=args.lambda_gate_smooth,
        )
        
        # Validate
        val_metrics = evaluate(model, valid_loader, device, gene_set_masks)
        
        scheduler.step()
        
        # Get current alpha
        alpha_val = model.geneset_alpha.item() if model.use_geneset_residual else 0.0
        
        # Log
        print(
            f"[Epoch {epoch:3d}] "
            f"Train: PCC={train_metrics['pcc']:.4f}, MSE={train_metrics['mse']:.4f}, Con={train_metrics['con']:.4f} | "
            f"Val: PCC={val_metrics['pcc']:.4f}, RMSE={val_metrics['rmse']:.4f} | "
            f"α={alpha_val:.4f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_pcc": train_metrics["pcc"],
            "train_mse": train_metrics["mse"],
            "train_con": train_metrics["con"],
            "val_pcc": val_metrics["pcc"],
            "val_rmse": val_metrics["rmse"],
            "geneset_alpha": alpha_val,
            "lr": scheduler.get_last_lr()[0],
        })
        
        # Save best
        if val_metrics["pcc"] > best_val_pcc:
            best_val_pcc = val_metrics["pcc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pcc": val_metrics["pcc"],
            }, run_dir / "best_model.pt")
    
    # ─── Test ─────────────────────────────────────────────────────────────────
    print("\n[Test] Loading best model...")
    ckpt = torch.load(run_dir / "best_model.pt")
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, device, gene_set_masks)
    
    # ─── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[Results] Best Epoch: {best_epoch}")
    print(f"[Results] Val  PCC: {best_val_pcc:.4f}")
    print(f"[Results] Test PCC: {test_metrics['pcc']:.4f}")
    print(f"[Results] Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"[Results] Final α: {model.geneset_alpha.item() if model.use_geneset_residual else 'N/A'}")
    print("=" * 70)
    
    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_pcc": best_val_pcc,
        "test_pcc": test_metrics["pcc"],
        "test_rmse": test_metrics["rmse"],
        "final_alpha": model.geneset_alpha.item() if model.use_geneset_residual else None,
        "args": vars(args),
        "history": history,
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Done] Results saved to {run_dir}")
    
    # ─── Interpretability Sample ──────────────────────────────────────────────
    print("\n[Interpretability] Sample analysis...")
    
    # Get a few samples
    sample_batch = next(iter(test_loader))
    sample_smiles = sample_batch["smiles"][:4]
    sample_expr = sample_batch["gene_expr"][:4].to(device)
    
    interp = model.get_interpretability(
        sample_smiles, sample_expr, gene_set_masks,
        gene_names=gene_columns,
        set_names=set_names,
        top_k=5,
    )
    
    print(f"\nGeneSet α = {interp['geneset_alpha']:.4f}")
    print("\nTop genes by gate (per drug):")
    for i, (smi, top_genes) in enumerate(zip(sample_smiles, interp.get("top_genes_by_gate", []))):
        genes_str = ", ".join([f"{g}({v:.2f})" for g, v in top_genes[:3]])
        print(f"  {smi[:30]:30s} → {genes_str}")
    
    print("\nTop gene sets (per drug):")
    for i, (smi, top_sets) in enumerate(zip(sample_smiles, interp.get("top_gene_sets", []))):
        sets_str = ", ".join([f"{s}({v:.2f})" for s, v in top_sets[:3]])
        print(f"  {smi[:30]:30s} → {sets_str}")


if __name__ == "__main__":
    main()