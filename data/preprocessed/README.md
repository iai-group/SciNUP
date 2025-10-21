This folder is empty because the files intended for it are too large to be included in the repository.

This folder is intended for the intermediate files between [raw data](../raw) and [SciNUP](../SciNUP) dataset.

[scripts/preprocess_data.py](../../scripts/preprocess_data.py) is used to preprocess JSON files from [../raw](), and save preprocessed data in this folder as CSV files:

- [arxiv-metadata.csv](.) -  contains 472,310 rows with columns: *['article_id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi',
'report-no', 'categories', 'license', 'abstract', 'versions',
'update_date', 'authors_parsed']*
- [citations.csv](.) - contains 734,454 rows with columns: *['article_id', 'references', 'num_references']*
- [authors.csv](.) - contains 243,985 rows with columns: *['author_id', 'authored_paper_ids', 'categories', 'author_name', 'num_papers']* 
- [candidate_users.csv](.) - a subset of [authors.csv](.) with users having >= *n_minimum_papers=5* papers. Contains 90,902 rows with columns: *['author_id', 'authored_paper_ids', 'categories', 'author_name', 'num_papers']*  