import pandas as pd
import json
from collections import Counter

# 파일 경로
ANNOTATION_PATH = "gdsc_drug_annotation.csv"
OUTPUT_PATH = "drug_pathway_mapping.json"

def main():
    # Annotation 파일 읽기
    df = pd.read_csv(ANNOTATION_PATH)
    print(f"[INFO] Loaded {len(df)} drugs from annotation")
    
    # DRUG_NAME -> TARGET_PATHWAY 매핑
    drug_to_pathway = {}
    for _, row in df.iterrows():
        drug_name = row['DRUG_NAME']
        pathway = row['TARGET_PATHWAY']
        drug_to_pathway[drug_name] = pathway
    
    # Pathway 분포 확인
    pathway_counts = Counter(drug_to_pathway.values())
    print("\n[INFO] Pathway distribution:")
    for pathway, count in pathway_counts.most_common(20):
        print(f"  {pathway}: {count}")
    
    # 제외할 pathway (너무 일반적이거나 분류 안 됨)
    exclude_pathways = {'Other', 'Unclassified', '', None}
    
    # 유효한 drug-pathway 매핑만 필터링
    valid_mapping = {
        drug: pathway 
        for drug, pathway in drug_to_pathway.items() 
        if pathway not in exclude_pathways and pd.notna(pathway)
    }
    
    print(f"\n[INFO] Valid drugs (excluding Other/Unclassified): {len(valid_mapping)}")
    
    # Pathway -> Drug list 역매핑
    pathway_to_drugs = {}
    for drug, pathway in valid_mapping.items():
        if pathway not in pathway_to_drugs:
            pathway_to_drugs[pathway] = []
        pathway_to_drugs[pathway].append(drug)
    
    # 최소 2개 drug가 있는 pathway만 사용 (contrastive에 필요)
    valid_pathways = {
        pathway: drugs 
        for pathway, drugs in pathway_to_drugs.items() 
        if len(drugs) >= 2
    }
    
    print(f"\n[INFO] Valid pathways (>=2 drugs): {len(valid_pathways)}")
    for pathway, drugs in sorted(valid_pathways.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  {pathway}: {len(drugs)} drugs")
    
    # 최종 매핑 (valid pathway에 속한 drug만)
    final_mapping = {
        drug: pathway 
        for drug, pathway in valid_mapping.items() 
        if pathway in valid_pathways
    }
    
    # 저장
    output = {
        'drug_to_pathway': final_mapping,
        'pathway_to_drugs': valid_pathways,
        'all_pathways': list(valid_pathways.keys()),
        'num_drugs': len(final_mapping),
        'num_pathways': len(valid_pathways),
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[DONE] Saved to {OUTPUT_PATH}")
    print(f"  - {len(final_mapping)} drugs")
    print(f"  - {len(valid_pathways)} pathways")


if __name__ == "__main__":
    main()