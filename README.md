# T-CLA: Taxonomy of Cultural-Linguistic Artifacts for Modeling Misalignment in NLP

T-CLA is a feature-level taxonomy for identifying and analyzing culturally situated linguistic variation in NLP. Developed from Kenyan Swahili, T-CLA captures observable linguistic artifacts that reflect language contact, lexical borrowing, ethnolinguistic influence, and culturally situated language use. The taxonomy provides a way to examine how these forms of variation relate to model behavior, including prediction error and subgroup fairness. T-CLA is introduced and empirically validated in our paper, **“T-CLA: Taxonomy of Cultural-Linguistic Artifacts for Modeling Misalignment in NLP,” accepted at EMNLP 2026**.

## Overview

T-CLA contains 170 cultural-linguistic artifacts (CLAs) organized into four higher-level categories:

- **Code-Mixing** — The integration of words or phrases from multiple languages within a single utterance, reflecting multilingual language use and language contact.

- **Loan Words** — Words borrowed from other languages and incorporated or adapted into Swahili, reflecting historical and ongoing lexical borrowing.

- **Sheng Usage** — Linguistic forms associated with Sheng, an urban Swahili-based sociolect characterized by lexical innovation and the mixing of Swahili, English, and other Kenyan languages.

- **Tribal Lexicons** — Lexical variation influenced by speakers’ ethnolinguistic backgrounds and local languages, capturing community-specific vocabulary, pronunciation, or spelling patterns.

These categories represent distinct but potentially co-occurring forms of sociolinguistic variation. T-CLA links these culturally grounded forms of language use to observable linguistic features, enabling their use in annotation, model evaluation, error analysis, and fairness analysis.


## Repository Contents

This repository contains the resources and code associated with the T-CLA framework and analyses presented in the paper.

```
📂 taxonomy/         # Contains human annotation data and identified CLAs
📂 code/             # Python scripts
📜 .gitattributes    # Git configuration file
📜 README.md         # This file
📜 poetry.lock       # Dependency lock file for reproducibility
📜 pyproject.toml    # Configuration for managing dependencies with Poetry
```

---
# Setup & Installation

This project uses **Poetry** for dependency management.

1. Download pipx: https://pipx.pypa.io/stable/installation/
2. Install poetry: https://python-poetry.org/docs/#installing-with-pipx
3. To generate figure 4:

```{python}
poetry run python code/confusion_matrices.py
```

To generate figure 5:

```{python}
poetry run python code/error_analysis.py
```


To generate figure 6:

```{python}
poetry run python code/subgroup_fairness_analysis.py
```

## Citation

If you use this work, please cite:


---
# Contributors

Kezia Oketch  
John Lalor      
Ahmed Abbasi    

# Acknowledgments

This research is supported by the University of Notre Dame's Human-centered Analytics Lab (HAL) and NSF.

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
