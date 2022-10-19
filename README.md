# Prioritizing Natural Language Test Cases Based on Highly-Used Game Features

This repository contains the source code of the experiments that we performed for our multi-objective optimization approach for prioritization of manual test cases. This work was submitted to the 45th International Conference on Software Engineering (ICSE) - Industry track (2023). 


---


Our approach was applied and evaluated with the data of the *Prodigy Math game* from our industry partner [Prodigy Education](https://www.prodigygame.com/main-en/). Our approach consists of two main steps: (1) automatic identification of the game feature(s) that are covered in manual test cases described in natural language and (2) prioritization of test cases based on the game features that are highly-used by players.

**(1) Zero-shot classification**: we used three techniques/models with strong zero-shot classification capabilities and different combinations (ensembles) of those techniques/models to automatically identify the covered game features from the textual description of test cases:

* BartLargeMNLI - [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
* CrossEncoderNLI - [cross-encoder/nli-distilroberta-base](https://huggingface.co/cross-encoder/nli-distilroberta-base)
* LatentEmb - [latent-embeddings](https://joeddav.github.io/blog/2020/05/29/ZSL.html)

**(2) Multi-objective optimization to prioritize the execution of test cases**: we performed two main experiments with the NSGA-II genetic algorithm to find optimal test case execution orderings:

* Without feature usage: our objectives functions are *number of covered game features* (maximize) and *test execution time* (minimize)
* With feature usage: our objectives functions are *number of covered highly-used game features* (maximize) and *test execution time* (minimize)


---


## Structure of directories
 
 The following directories contains the source code of all the approaches that were part of our experiments. 

 - [zero-shot-classification-experiments](/zero-shot-classification-experiments/): contains the notebook and scripts with the source code of our experiments with different zero-shot techniques for text classifications and their combinations.
 
 - [multi-objective-optimization-experiments](/multi-objective-optimization-experiments/): contains the notebooks and scripts with the source code of our experiments with different optimization approaches.
 
 
---


## Dependencies

The following dependencies are required to run the notebooks and scripts on your local machine:

- Python 3.9


 - [Transformers 4.21.2](https://huggingface.co/transformers/)

    `
    pip install transformers
    `


 - [Torch 1.10.0+cu113](https://pytorch.org/)

    `
    pip3 install -f torch torchvision
    `
    
    
 - [SentenceTransformers 2.2.2](https://www.sbert.net/)

    `
    pip install sentence-transformers
    `
    
    
 - [Scikit-learn 1.1.2](https://scikit-learn.org/stable/)

    `
    pip install scikit-learn
    `
    

 - [Scipy 1.9.1](https://scipy.org/)

    `
    pip install scipy
    `


- [cliffs_delta 1.0.0](https://github.com/neilernst/cliffsDelta)

    `
    pip install cliffs-delta
    `
    

 - [NLTK 3.4.1](https://www.nltk.org/)

    `
    pip install nltk
    `


 - [fasttext 0.9.2](https://fasttext.cc/)

    `
    pip install fasttext
    `
    
    
 - [Numpy 1.22.4](https://numpy.org/)

    `
    pip install numpy
    `


 - [Pandas 1.4.3](https://pandas.pydata.org/)
 
    `
    pip install pandas
    `


 - [Matplotlib 3.5.3](https://matplotlib.org/)

    `
    pip install matplotlib
    `


 - [Plotly 5.10.0](https://plotly.com/)

    `
    pip install plotly
    `


 - [Seaborn 0.12.0](https://seaborn.pydata.org/index.html)

    `
    pip install seaborn
    `
    
    
- [MLFlow 1.29.0](https://mlflow.org/)

    `
    pip install mlflow
    `

Any questions about the work can be sent to the first author by email (markosviggiato [at] gmail.com)
