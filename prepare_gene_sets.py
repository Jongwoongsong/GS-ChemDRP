import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def parse_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
    """
    Parse GMT (Gene Matrix Transposed) file format
    
    GMT format (tab-separated):
    SET_NAME    DESCRIPTION    GENE1    GENE2    GENE3    ...
    """
    gene_sets = {}
    
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            set_name = parts[0]
            # parts[1] is description, skip
            genes = parts[2:]
            
            gene_sets[set_name] = genes
    
    print(f"[GMT] Loaded {len(gene_sets)} gene sets from {gmt_path}")
    return gene_sets


def filter_gene_sets_by_overlap(
    gene_sets: Dict[str, List[str]],
    target_genes: List[str],
    min_overlap: int = 3,
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Filter gene sets to only include genes in target_genes
    and remove sets with too few overlapping genes
    """
    target_set = set(str(g) for g in target_genes)
    filtered = {}
    overlap_counts = {}
    
    for name, genes in gene_sets.items():
        overlap = [g for g in genes if str(g) in target_set]
        overlap_counts[name] = len(overlap)
        
        if len(overlap) >= min_overlap:
            filtered[name] = overlap
    
    print(f"[Filter] {len(filtered)}/{len(gene_sets)} sets have >= {min_overlap} overlapping genes")
    return filtered, overlap_counts


def create_kegg_based_gene_sets(
    kegg_dir: str,
    target_genes: List[str],
) -> Dict[str, List[str]]:
    """
    Create gene sets from KEGG pathway XML files
    (Uses existing KEGG pathway parsing)
    """
    from xml.etree import ElementTree as ET
    
    target_set = set(str(g) for g in target_genes)
    gene_sets = {}
    
    for xml_file in Path(kegg_dir).glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            pathway_name = root.get('title', xml_file.stem)
            genes = []
            
            for entry in root.findall('.//entry[@type="gene"]'):
                gene_ids = entry.get('name', '').replace('hsa:', '').split()
                for gid in gene_ids:
                    if gid in target_set:
                        genes.append(gid)
            
            genes = list(set(genes))  # unique
            if len(genes) >= 3:
                gene_sets[pathway_name] = genes
                
        except Exception as e:
            print(f"[WARN] Failed to parse {xml_file}: {e}")
    
    print(f"[KEGG] Created {len(gene_sets)} gene sets from KEGG pathways")
    return gene_sets


def create_go_cluster_gene_sets(
    go_sim_path: str,
    gene_list: List[str],
    num_clusters: int = 50,
) -> Dict[str, List[str]]:
    """
    Create gene sets by clustering based on GO similarity
    """
    from sklearn.cluster import SpectralClustering
    
    go_sim = np.load(go_sim_path)
    
    # Ensure valid similarity matrix
    go_sim = np.nan_to_num(go_sim, nan=0.0)
    go_sim = (go_sim + go_sim.T) / 2  # symmetrize
    np.fill_diagonal(go_sim, 1.0)
    
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=42,
    )
    labels = clustering.fit_predict(go_sim)
    
    gene_sets = {}
    for c in range(num_clusters):
        cluster_idx = np.where(labels == c)[0]
        genes = [str(gene_list[i]) for i in cluster_idx]
        if len(genes) >= 3:
            gene_sets[f"GO_CLUSTER_{c:02d}"] = genes
    
    print(f"[GO Cluster] Created {len(gene_sets)} gene sets")
    return gene_sets


def print_gene_set_stats(gene_sets: Dict[str, List[str]]):
    """Print statistics about gene sets"""
    sizes = [len(genes) for genes in gene_sets.values()]
    
    print(f"\n[Gene Set Statistics]")
    print(f"  Total sets: {len(gene_sets)}")
    print(f"  Genes per set: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    print(f"  Total unique genes: {len(set(g for genes in gene_sets.values() for g in genes))}")
    
    print(f"\n  Top 10 sets by size:")
    sorted_sets = sorted(gene_sets.items(), key=lambda x: len(x[1]), reverse=True)
    for name, genes in sorted_sets[:10]:
        print(f"    {name}: {len(genes)} genes")


def main():
    parser = argparse.ArgumentParser(description="Prepare gene sets for GS model")
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["gmt", "kegg", "go_cluster"],
                       help="Gene set source: gmt (MSigDB), kegg, or go_cluster")
    
    # Input paths
    parser.add_argument("--gmt_path", type=str, default=None,
                       help="Path to GMT file (for mode=gmt)")
    parser.add_argument("--kegg_dir", type=str, default=None,
                       help="Path to KEGG XML directory (for mode=kegg)")
    parser.add_argument("--go_path", type=str, default=None,
                       help="Path to GO similarity matrix (for mode=go_cluster)")
    
    # Target genes
    parser.add_argument("--gene_list_path", type=str, required=True,
                       help="Path to CSV with gene columns, or text file with gene IDs")
    
    # Clustering options
    parser.add_argument("--num_clusters", type=int, default=50,
                       help="Number of clusters for GO clustering")
    
    # Filter options
    parser.add_argument("--min_overlap", type=int, default=3,
                       help="Minimum genes per set")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    # Load target gene list
    if args.gene_list_path.endswith('.csv'):
        df = pd.read_csv(args.gene_list_path, nrows=1)
        gene_list = [c for c in df.columns if c.isdigit()]
    else:
        with open(args.gene_list_path, 'r') as f:
            gene_list = [line.strip() for line in f if line.strip().isdigit()]
    
    print(f"[Genes] Loaded {len(gene_list)} target genes")
    
    # Create gene sets
    if args.mode == "gmt":
        if not args.gmt_path:
            raise ValueError("--gmt_path required for mode=gmt")
        gene_sets = parse_gmt_file(args.gmt_path)
        gene_sets, _ = filter_gene_sets_by_overlap(gene_sets, gene_list, args.min_overlap)
    
    elif args.mode == "kegg":
        if not args.kegg_dir:
            raise ValueError("--kegg_dir required for mode=kegg")
        gene_sets = create_kegg_based_gene_sets(args.kegg_dir, gene_list)
    
    elif args.mode == "go_cluster":
        if not args.go_path:
            raise ValueError("--go_path required for mode=go_cluster")
        gene_sets = create_go_cluster_gene_sets(args.go_path, gene_list, args.num_clusters)
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Print stats
    print_gene_set_stats(gene_sets)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(gene_sets, f, indent=2)
    
    print(f"\n[Done] Saved to {args.output}")