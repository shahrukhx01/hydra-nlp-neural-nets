## Introduction:
A (http://hydra.cc/)[hydra] based Natural Language Processing (NLP) pipeline boilerplate with loggers embbeded, allowing for streamlining deep learning models and accomodating experimentation while also being able to write modular scalable code. The best feature is that the code is completely parametrized via config file, which minimizes code changes when data changes etc.

### Features Implemented till now:
- DataLoaders:
    - Elasticsearch (read and write)
    - Kafka (reading from queues and write to Elasticsearch index)
    - Eland (Elasticsearch to pandas direct)
    - Service Now (Service now to Elasticsearch index)
- Prepprocessing:
    - Tokenization
    - Stopword removal 
    - Lemmatization
    - Token Cleanser
    
- Models:
    - Any (Pytorch/TensorFlow Neural Net Architecture)
    
- Futher configurations and code extension can be added based on use case.
