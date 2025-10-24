# Data

Here is the high-level description of each folder under `data/`:

- [SciNUP](SciNUP) - dataset components.
- [raw](raw) - arXiv dataset.
- [preprocessed](preprocessed) - intermediate files between [raw](raw) and [SciNUP](SciNUP)
- [retrieval_results](retrieval_results) - groud truth qrels and trec files from different retrieval methods.
- [docs](docs) - authors folder containing {author_id}.jsonl files with all the candidate items for that author_id. Used for indexing and retrieval. Can be downloaded from [here](https://iai.group/downloads/SciNUP/authors.tgz).
- [indexes](indexes) - indexes created for sparse ([download](https://iai.group/downloads/SciNUP/sparse.tgz)) (BM25 and RM3) and dense ([download](https://iai.group/downloads/SciNUP/dense.tgz)) (SciBERT) retrieval.