# SciNUP: Natural Language User Interest Profiles for Scientific Literature Recommendation

This repository provides resources developed within the following article [PDF]:

## Summary

The use of natural language (NL) user profiles in recommender systems offers greater transparency and user control compared to traditional representations. 
However, there is scarcity of large-scale, publicly available test collections for evaluating NL profile-based recommendation. 
To address this gap, we introduce SciNUP, a novel synthetic dataset for scholarly recommendation that leverages authors' publication histories to generate NL profiles and corresponding ground truth items. We use this dataset to conduct a comparison of baseline methods, ranging from sparse and dense retrieval approaches to state-of-the-art LLM-based rerankers.
Our results show that while baseline methods achieve comparable performance, they often retrieve different items, indicating complementary behaviors. At the same time, considerable headroom for improvement remains, highlighting the need for effective NL-based recommendation approaches.
The SciNUP dataset thus serves as a valuable resource for fostering future research and development in this area.

## SciNUP Dataset

The SciNUP dataset contains NL profiles of 1000 researchers for scientific literature recommendation with the following components:

- NL profiles and candidate items: [data/SciNUP/dataset.jsonl]()
- Ground truth items: [data/SciNUP/sampled_users.jsonl]()
- NL profile breadth categorization: [data/SciNUP/breadth_classification.tsv]()

| Attribute | SciNUP |
|-----------|--------|
| #Authors | 1,000 |
| #Authored papers (min / median / max) | 10 / 20 / 260 |
| #Candidate items per author | 1,000 |
| #Ground truth papers per author (min / median / max) | 1 / 27 / 438 |
| Profile length (words) | 117 ± 55 |
| #Narrow / Medium / Broad NL profiles | 679 / 256 / 65 |

Details about dataset creation steps can be found under [data/SciNUP/README.md]()

## Benchmarking

We benchmarked Sparse (BM25, RM3), Dense (kNN-SciBERT, BGE-large, BGE-v2-M3, BGE-v2-MiniCPM) and LLM-reranking (PRP-Llama-3-8B, PRP-Llama-3.3-70B, PRP-GPT-4o-mini) methods on our dataset. Additionally, we evaluated ensemble model using reciprocal rank fusion to fuse best-performing results in each category (RM3, BGE-v2-MiniCPM and PRP-GPT-4o-mini). 


| Model                  | R@100   | MAP        | MRR        | NDCG@10   | Runfile
|------------------------|---------|------------|------------|-----------|-----------|
| BM25                   | 0.3491  | 0.1148     | 0.4661     | 0.2869    | [data/retrieval_results/bm25.trec]() |
| RM3                    | 0.3570  | 0.1391     | 0.5147     | 0.3251    | [data/retrieval_results/rm3.trec]() |
| kNN-SciBERT            | 0.1480  | 0.0232     | 0.2182     | 0.1019    | [data/retrieval_results/knn_scibert.trec]()  |
| BGE-Large              | 0.2826  | 0.0783     | 0.3666     | 0.2072    | [data/retrieval_results/bge_large.trec]()  |
| BGE-v2-M3              | 0.3472  | 0.1152     | 0.4633     | 0.2763    | [data/retrieval_results/bge_v2_m3.trec]()  |
| BGE-v2-MiniCPM         | **0.4203** | 0.1673     | 0.5393     | 0.3541    | [data/retrieval_results/bge_v2_minicpm.trec]()  |
| PRP-Llama-3 (8B)       | 0.3491  | 0.1165     | 0.4774     | 0.2925    | [data/retrieval_results/prp_llama_8b.trec]()  |
| PRP-Llama-3.3 (70B)    | 0.3491  | 0.1423     | 0.5378     | 0.3541    | [data/retrieval_results/prp_llama_70b.trec]()  |
| PRP-GPT-4o-mini        | 0.3491  | 0.1405     | 0.5297     |   0.3542  | [data/retrieval_results/prp_gpt.trec]()  |
| Ensemble               | 0.4136  | **0.2163** | **0.6333** | **0.4481** | [data/retrieval_results/rrf_fused.trec]()  |

Implementation of these methods can be found under [SciNUP/models]() and [scripts/](). The main script to run retrieval including the sample usage is under [scripts/run_retrieval.py]

The numbers shown in the table above are generated using `trec_eval`:

```
trec_eval -m recall.100 -m map -m recip_rank -m ndcg_cut.10 data/retrieval_results/ground_truth_qrels.txt PATH_TO_DESIRED_RUNFILE
```

## Citation

If you use the resources presented in this repository, please cite:

```
TBD
```

## Contact

Should you have any questions, please contact Mariam Arustashvili at mariam.arustashvili[AT]uis.no (with [AT] replaced by @).
