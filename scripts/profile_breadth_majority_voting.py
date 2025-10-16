import collections

# Input files
files = [
    "nl_profiles_classification/nl_profiles_classification_gemini-2.5-flash.tsv",
    "nl_profiles_classification/nl_profiles_classification_gpt4o.tsv",
    "nl_profiles_classification/nl_profiles_classifications_llama3.3:70b.tsv",
]

# Output file
output_file = "data/SciNUP/breadth_classification.tsv"

votes = collections.defaultdict(list)

for f in files:
    with open(f, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            author_id, label = line.split("\t")
            votes[author_id].append(label)

with open(output_file, "w", encoding="utf-8") as outfile:
    for author_id, labels in votes.items():
        counter = collections.Counter(labels)

        if len(counter) == 3:  # all three different
            majority_label = "Medium"  # default to Medium
        else:
            majority_label = counter.most_common(1)[0][0]

        outfile.write(f"{author_id}\t{majority_label}\n")

print(f"Majority vote classifications written to {output_file}")
