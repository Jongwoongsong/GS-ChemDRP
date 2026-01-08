import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import math


class DrugConditionedGeneGating(nn.Module):
    """
    Drug-conditioned gene gating layer.
    For each gene i: gate_i(d) = σ(w_i^T d + b_i)
    Output: x_gated = gate * x
    """
    def __init__(
        self, 
        drug_dim: int, 
        num_genes: int, 
        hidden_dim: int = 256,
        use_residual: bool = True,
        gate_bias_init: float = 1.0,  # Start with gates ~open
    ):
        super().__init__()
        self.num_genes = num_genes
        self.use_residual = use_residual
        
        # MLP: drug_emb -> gate values for each gene
        self.gate_mlp = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_genes),
        )
        
        # Initialize bias so gates start ~open (sigmoid(1) ≈ 0.73)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)
        
        # Learnable residual scale (start small)
        if use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, drug_emb: torch.Tensor, gene_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            drug_emb: [B, D] drug embedding
            gene_expr: [B, G] gene expression values
        Returns:
            gated_expr: [B, G] gated gene expression
            gate_values: [B, G] gate values (for analysis/regularization)
        """
        # Compute gates
        gate_logits = self.gate_mlp(drug_emb)  # [B, G]
        gate_values = torch.sigmoid(gate_logits)  # [B, G], range (0, 1)
        
        # Apply gating
        if self.use_residual:
            # Residual: x_out = x + scale * (gate - 0.5) * x
            # This way, gate=0.5 means no change, gate>0.5 amplifies, gate<0.5 suppresses
            gated_expr = gene_expr + self.residual_scale * (gate_values - 0.5) * gene_expr
        else:
            # Direct: x_out = gate * x
            gated_expr = gate_values * gene_expr
        
        return gated_expr, gate_values
    
    def get_gate_l1_loss(self, gate_values: torch.Tensor) -> torch.Tensor:
        """L1 regularization to encourage sparse gene selection"""
        return gate_values.abs().mean()
    
    def get_gate_smoothness_loss(
        self, 
        gate_values: torch.Tensor, 
        gene_set_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage genes in same set to have similar gates.
        gene_set_masks: [num_sets, num_genes] binary masks
        """
        loss = 0.0
        num_sets = gene_set_masks.size(0)
        
        for s in range(num_sets):
            mask = gene_set_masks[s]  # [G]
            if mask.sum() < 2:
                continue
            
            # Get gates for genes in this set
            gates_in_set = gate_values[:, mask.bool()]  # [B, set_size]
            
            # Variance within set (want this to be small)
            set_var = gates_in_set.var(dim=1).mean()
            loss = loss + set_var
        
        return loss / max(num_sets, 1)


class GeneSetEncoder(nn.Module):
    """Encode genes into gene set representations"""
    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Gene embedding (learnable)
        self.gene_embed = nn.Parameter(torch.randn(num_genes, embed_dim) * 0.02)
        
        # Project expression-scaled embeddings
        self.gene_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Intra-set attention (genes within same set attend to each other)
        self.intra_set_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.intra_set_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        gene_expr: torch.Tensor, 
        gene_set_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gene_expr: [B, G] gene expression
            gene_set_masks: [S, G] binary masks for each gene set
        Returns:
            set_embeds: [B, S, D] gene set embeddings
        """
        B = gene_expr.size(0)
        S, G = gene_set_masks.shape
        
        # Scale gene embeddings by expression: [B, G, D]
        gene_embeds = self.gene_embed.unsqueeze(0) * gene_expr.unsqueeze(-1)
        gene_embeds = self.gene_proj(gene_embeds)  # [B, G, D]
        
        # Aggregate genes into sets
        set_embeds_list = []
        for s in range(S):
            mask = gene_set_masks[s]  # [G]
            genes_in_set = gene_embeds[:, mask.bool(), :]  # [B, set_size, D]
            
            if genes_in_set.size(1) == 0:
                # Empty set -> zero embedding
                set_embed = torch.zeros(B, self.embed_dim, device=gene_expr.device)
            else:
                # Intra-set attention
                attn_out, _ = self.intra_set_attn(
                    genes_in_set, genes_in_set, genes_in_set
                )
                attn_out = self.intra_set_norm(genes_in_set + attn_out)
                
                # Mean pooling
                set_embed = attn_out.mean(dim=1)  # [B, D]
            
            set_embeds_list.append(set_embed)
        
        set_embeds = torch.stack(set_embeds_list, dim=1)  # [B, S, D]
        return set_embeds


class DrugGeneSetCrossAttention(nn.Module):
    """Drug queries attend to gene sets"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        drug_emb: torch.Tensor, 
        set_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            drug_emb: [B, D] drug embedding
            set_embeds: [B, S, D] gene set embeddings
        Returns:
            cell_repr: [B, D] drug-aware cell representation
            attn_weights: [B, S] attention over gene sets
        """
        # Drug as query
        drug_q = drug_emb.unsqueeze(1)  # [B, 1, D]
        
        # Cross attention
        attn_out, attn_weights = self.cross_attn(
            drug_q, set_embeds, set_embeds
        )  # attn_out: [B, 1, D], attn_weights: [B, 1, S]
        
        # Residual + FFN
        attn_out = self.norm(drug_q + attn_out)
        ffn_out = self.ffn(attn_out)
        out = self.ffn_norm(attn_out + ffn_out)
        
        cell_repr = out.squeeze(1)  # [B, D]
        attn_weights = attn_weights.squeeze(1)  # [B, S]
        
        return cell_repr, attn_weights


class BaseCellEncoder(nn.Module):
    """Simple MLP cell encoder (baseline)"""
    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        return self.encoder(gene_expr)


class GSGeneSetGatingModel(nn.Module):
    """
    Full model with:
    1. Drug-conditioned Gene Gating
    2. Base MLP cell encoder
    3. Gene Set Cross-Attention (residual branch)
    4. Prediction head
    """
    def __init__(
        self,
        num_genes: int,
        num_gene_sets: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        drug_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        use_gene_gating: bool = True,
        use_geneset_residual: bool = True,
        geneset_residual_init: float = 0.1,  # Start small
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_gene_sets = num_gene_sets
        self.embed_dim = embed_dim
        self.use_gene_gating = use_gene_gating
        self.use_geneset_residual = use_geneset_residual
        
        # Drug encoder (ChemBERTa)
        self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        self.drug_encoder = AutoModel.from_pretrained(drug_model_name)
        drug_dim = self.drug_encoder.config.hidden_size  # 768
        
        # Freeze drug encoder initially
        for param in self.drug_encoder.parameters():
            param.requires_grad = False
        
        # Drug projection
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # 1. Drug-conditioned Gene Gating
        if use_gene_gating:
            self.gene_gating = DrugConditionedGeneGating(
                drug_dim=embed_dim,
                num_genes=num_genes,
                hidden_dim=embed_dim,
                use_residual=True,
            )
        
        # 2. Base cell encoder (MLP)
        self.base_cell_encoder = BaseCellEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # 3. Gene Set encoder + Cross-Attention (residual branch)
        if use_geneset_residual:
            self.geneset_encoder = GeneSetEncoder(
                num_genes=num_genes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.geneset_cross_attn = DrugGeneSetCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            # Learnable residual weight (start small)
            self.geneset_alpha = nn.Parameter(torch.tensor(geneset_residual_init))
        
        # 4. Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
    
    def encode_drug(self, smiles: List[str]) -> torch.Tensor:
        """Encode SMILES strings to drug embeddings"""
        device = next(self.parameters()).device
        
        inputs = self.drug_tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = self.drug_encoder(**inputs)
        
        drug_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        drug_emb = self.drug_proj(drug_emb)
        
        return drug_emb
    
    def forward(
        self,
        smiles: List[str],
        gene_expr: torch.Tensor,
        gene_set_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            smiles: List of SMILES strings [B]
            gene_expr: [B, G] gene expression
            gene_set_masks: [S, G] gene set masks
        Returns:
            Dict with 'pred', 'gate_values', 'set_attn', etc.
        """
        # 1. Encode drug
        drug_emb = self.encode_drug(smiles)  # [B, D]
        
        # 2. Gene gating (if enabled)
        gate_values = None
        if self.use_gene_gating:
            gated_expr, gate_values = self.gene_gating(drug_emb, gene_expr)
        else:
            gated_expr = gene_expr
        
        # 3. Base cell encoder
        c_base = self.base_cell_encoder(gated_expr)  # [B, D]
        
        # 4. Gene Set branch (residual)
        set_attn = None
        if self.use_geneset_residual:
            set_embeds = self.geneset_encoder(gated_expr, gene_set_masks)  # [B, S, D]
            c_geneset, set_attn = self.geneset_cross_attn(drug_emb, set_embeds)  # [B, D], [B, S]
            
            # Residual combination
            cell_repr = c_base + self.geneset_alpha * c_geneset
        else:
            cell_repr = c_base
        
        # 5. Predict
        combined = torch.cat([drug_emb, cell_repr], dim=-1)  # [B, 2D]
        pred = self.predictor(combined).squeeze(-1)  # [B]
        
        return {
            "pred": pred,
            "drug_emb": drug_emb,
            "cell_repr": cell_repr,
            "gate_values": gate_values,
            "set_attn": set_attn,
            "geneset_alpha": self.geneset_alpha if self.use_geneset_residual else None,
        }
    
    def get_auxiliary_losses(
        self,
        gate_values: Optional[torch.Tensor],
        gene_set_masks: torch.Tensor,
        lambda_l1: float = 0.01,
        lambda_smooth: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for gene gating"""
        losses = {}
        
        if gate_values is not None and self.use_gene_gating:
            # L1 sparsity
            losses["gate_l1"] = lambda_l1 * self.gene_gating.get_gate_l1_loss(gate_values)
            
            # Smoothness within gene sets
            losses["gate_smooth"] = lambda_smooth * self.gene_gating.get_gate_smoothness_loss(
                gate_values, gene_set_masks
            )
        
        return losses
    
    def get_interpretability(
        self,
        smiles: List[str],
        gene_expr: torch.Tensor,
        gene_set_masks: torch.Tensor,
        gene_names: List[str],
        set_names: List[str],
        top_k: int = 10,
    ) -> Dict:
        """Get interpretable outputs"""
        self.eval()
        with torch.no_grad():
            out = self.forward(smiles, gene_expr, gene_set_masks)
        
        results = {
            "predictions": out["pred"].cpu().numpy(),
            "geneset_alpha": out["geneset_alpha"].item() if out["geneset_alpha"] is not None else None,
        }
        
        # Top genes by gate value
        if out["gate_values"] is not None:
            gate_vals = out["gate_values"].cpu().numpy()  # [B, G]
            top_genes = []
            for b in range(len(smiles)):
                sorted_idx = gate_vals[b].argsort()[::-1][:top_k]
                top_genes.append([
                    (gene_names[i], gate_vals[b, i]) for i in sorted_idx
                ])
            results["top_genes_by_gate"] = top_genes
        
        # Top gene sets by attention
        if out["set_attn"] is not None:
            set_attn = out["set_attn"].cpu().numpy()  # [B, S]
            top_sets = []
            for b in range(len(smiles)):
                sorted_idx = set_attn[b].argsort()[::-1][:top_k]
                top_sets.append([
                    (set_names[i], set_attn[b, i]) for i in sorted_idx
                ])
            results["top_gene_sets"] = top_sets
        
        return results


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing GSGeneSetGatingModel...")
    
    # Dummy data
    B, G, S = 4, 949, 48
    gene_expr = torch.randn(B, G)
    gene_set_masks = torch.zeros(S, G)
    for s in range(S):
        # Random genes in each set
        idx = torch.randperm(G)[:20]
        gene_set_masks[s, idx] = 1
    
    smiles = ["CCO", "CCN", "CCC", "CCCC"]
    
    # Model
    model = GSGeneSetGatingModel(
        num_genes=G,
        num_gene_sets=S,
        embed_dim=256,
        use_gene_gating=True,
        use_geneset_residual=True,
    )
    
    print(f"Trainable params: {count_parameters(model):,}")
    
    # Forward
    out = model(smiles, gene_expr, gene_set_masks)
    print(f"Prediction shape: {out['pred'].shape}")
    print(f"Gate values shape: {out['gate_values'].shape}")
    print(f"Set attention shape: {out['set_attn'].shape}")
    print(f"GeneSet alpha: {out['geneset_alpha'].item():.4f}")
    
    # Auxiliary losses
    aux_losses = model.get_auxiliary_losses(out["gate_values"], gene_set_masks)
    print(f"Aux losses: {aux_losses}")
    
    print("✅ Test passed!")