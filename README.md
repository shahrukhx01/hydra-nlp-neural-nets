## Text Similarity Component:

### Methods tried:
- Fuzzy matching:
    - High scores for even a single word match and amplified discrepancy in case of differing lengths of change and incident texts
    - Implemented penalization of difference in lengths by reducing the similarity score when the texts have huge difference in length. 
- TF-IDF:
    - Using TF-IDF to pick out the main entities and computing similarity between these entities using cosine similarity had already been implemented
    - Additionaly the top TF-IDF entities computed can contribute towards domain significant words that can be considered for special case handling in similarity metric computations
- nGrams:
    - Extracting sets of ngrams from change and incident text was performed for n = 2 to 7
    - With both cosine similarity and jaccard distance, the similarity computation was unbeneficial since change and incident texts don't seem to have common ngrams often. Hence the intersection of the sets is almost always empty
- Embeddings:
    - Embedding models used:
        - Word2Vec Customized
        - FastText Customized
    - Word level:
        - FastText most suitable for this to enable OOV and character level handling in case not customized
        - Word level similarity / distance computation allows us to apply domain specific or batch specific scoring mechanisms
    - Sentence level:
        - Sentence level word mover's distance computation is benificial in the case of handling varying length texts
        - However, currently the topic modeling is not very clear at the sentence level and hence similarity is hardly found
    - Hybrid:
        - Combination of sentence level and word level
            - Usage of embedding model across granularity:
                - Trying word2vec for sentence level embedding gave infinite similarity with earth mover's distance
                - Currently FastText is used for both sentence level and word level
