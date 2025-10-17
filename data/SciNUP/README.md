# SciNUP dataset creation

Four main steps lead to SciNUP dataset:

1. Preprocessing arXiv dataset
2. Sampling authors
3. Dataset generation
4. Extending dataset


## 1. Preprocessing arXiv dataset

[scripts/preprocess_data.py](scripts/preprocess_data.py) is used to preprocess JSON files from [data/raw](../raw/), and save preprocessed data as CSV files.

  - input:
    + Initial data from ArXiv dataset - three JSON files stored in [data/raw](../raw/)
  - output:
    + Four CSV files stored in [data/preprocessed](../preprocessed/):
        - [arxiv-metadata.csv](../preprocessed/arxiv-metadata.csv) -  contains 472,310 rows with columns: *['article_id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi',
            'report-no', 'categories', 'license', 'abstract', 'versions',
            'update_date', 'authors_parsed']*
        - [citations.csv](../preprocessed/) - contains 734,454 rows with columns: *['article_id', 'references', 'num_references']*
        - [authors.csv](../preprocessed/) - contains 243,985 rows with columns: *['author_id', 'authored_paper_ids', 'categories', 'author_name', 'num_papers']* 
        - [candidate_users.csv](../preprocessed/) - a subset of [authors.csv](../preprocessed/) with users having >= *n_minimum_papers=5* papers. Contains 90,902 rows with columns: *['author_id', 'authored_paper_ids', 'categories', 'author_name', 'num_papers']* 

## 2. Sampling authors

  - input:
    + [data/preprocessed/candidate_users.csv](../preprocessed)
    + [data/preprocessed/arxiv-metadata.csv](../preprocessed)
    + [data/preprocessed/citations.csv](../preprocessed)
  - output:
    + [data/SciNUP/sampled_users.jsonl](.) - contains the following four fields: *author_id, author_name, nl_profile_input, ground_truth_items*.

[scripts/sample_users.py](../../scripts/sample_users.py) samples 1000 users, transforms them into Author objects ([src/components/author.py](../../src/components/author.py)) and saves as a JSONL file.

After this step, one component of the SciNUP dataset, ground truth items, is created.

## 3. Generate dataset

Ground truth items are already extracted and stored in the sampling step. Remaining two major components - NL profiles and candidate items set are generated in this step.



  - input:
    + [data/preprocessed/candidate_users.csv](../preprocessed/)
    + [data/preprocessed/arxiv-metadata.csv](../preprocessed/)
    + [data/SciNUP/sampled_users.jsonl](.)
  - output:
    + [data/SciNUP/dataset.jsonl](.)

[scripts/generate_dataset.py](../../scripts/generate_dataset.py) generates a dataset that contains NL profiles and candidate item sets for the 1000 sampled users.

## 4. Extend dataset

This step extends dataset with NL profile breadth category. 

NL profiles, which characterize a research's interests, can vary significantly in breadth. Some researchers have a wider scope of interests, while others focus on a more narrow topic. These differences can influence the performance of recommendation methods. 
To account for this, we automatically classify each profile's breadth and provide this classification as metadata. Specifically, we use a few-shot prompt and take the majority vote from three distinct LLMs (Llama-70B, Gemini-2.5-Flash, GPT-4o) to ensure a robust classification that mitigates potential single-model biases.

[scripts/classify_nl_profiles.py](../../scripts/classify_nl_profiles.py) prompts an LLM to classify natural language profiles into narrow,
medium, or broad categories. Sample usage is available in the docstring of the script.  

Majority voting is done by running [scripts/profile_breadth_majority_voting.py](../../scripts/profile_breadth_majority_voting.py) and results in a [data/SciNUP/breadth_classification.tsv](breadth_classification.tsv) file.