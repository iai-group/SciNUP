# Benchmarking

All runnable files are under [../scripts/](../scripts/). Everything needs to be run from the project root directory. 

The following methods are benchmarked on the SciNUP dataset:

- BM25 (sparse)
- Query Likelihood “RM3” (sparse)
- SciBERT + kNN (dense)
- BAAI BGE rerankers
    + bge-reranker-large
    + bge-reranker-v2-m3
    + bge-reranker-v2-minicpm-layerwise
- LLM pairwise reranking
    + GPT-4o-mini
    + Llama-70B
    + Llama-8B


For BM25, RM3 and kNN-SciBERT, indexes are built before running retrieval.

Before building indexes, [../scripts/articles_to_docs.py](../scripts/articles_to_docs.py) is used to format the candidate items sets. Output is saved in [../data/docs/authors](../data/docs/authors), a single JSONL file for each author `{author_id}/docs.jsonl`


## Building indexes

- input: `../data/docs/authors/{author_id}/docs.jsonl`
- output: `../data/indexes/{index_type}/{index_name}/authors/{author_id}/`

Scripts:

- [../scripts/create_sparse_index.py](../scripts/create_sparse_index.py)
- [../scripts/build_scibert_index.py](../scripts/build_scibert_index.py)

## Retrieval

Retrieval is done based on indexes (BM25, RM3, kNN-SciBERT) or directly based on the query-document scoring (BGE rerankers). 

- input: 
    + `../data/indexes/{index_type}/{index_name}/authors/{author_id}/` for sparse methods and SciBERT-KNN
    + `../data/docs/authors/{author_id}/docs.jsonl` for BGE methods.
- output: `data/retrieval_results/{retrieval_method}.trec`

Scripts:

- [../scripts/run_retrieval.py](../scripts/run_retrieval.py)
- [../scripts/run_bge_retrieval.py](../scripts/run_bge_retrieval.py)

## LLM Reranking

Script: [../scripts/run_llm_reranking.py]

Example usage:
```
    python run_llm_reranking.py \
    --retrieval_results_path input_path \
    --output_file output_path \
    --llm_model_name model_name \
```
