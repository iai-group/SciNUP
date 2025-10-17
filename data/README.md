# Data

Here is the high-level description of each folder under `data/`:

- [SciNUP]() - dataset components
- [raw]() - arXiv dataset
- [preprocessed]() - intermediate files between [raw]() and [SciNUP]()
- [retrieval_results]() - groud truth qrels and trec files from different retrieval methods
- [docs/authors]() - {author_id}.jsonl files with all the candidate items for that author_id. Used for indexing and retrieval.
- [indexes]() - indexes created for [sparse](indexes/sparse) (BM25 and RM3) and [dense](indexes/dense/scibert) (SciBERT) retrieval.