# Open-Domain Question Answering
## 1️⃣ Introduction
This project handles the Open-Domain Question Answering (ODQA) task, which aims to answer questions based on world knowledge when no specific passage is given. <br> The model consists of two stages: the "retriever" stage, which finds relevant documents related to the question, and the "reader" stage, which reads and identifies appropriate answers within the retrieved documents.
<p align="center"><img src="https://user-images.githubusercontent.com/65378914/217729308-057c696b-6c1f-41eb-970e-14ea6281c67c.png" width="80%" height="80%"/></p>
 - Evaluation metrics: Exact Match (EM), F1 Score <br>
<details>
    <summary><b><font size="10">Project Tree</font></b></summary>
<div markdown="1">

```
.
├─ README.md
├─ assets
│  ├─ mrc.png
│  └─ odqa.png
├─ inference.py # for testing
├─ install
│  └─ install_requirements.sh
├─ mlm.py
├─ model
│  └─ Retrieval
│     ├─ BertEncoder.py
│     └─ RobertaEncoder.py
├─ mrc.py
├─ notebooks
│  ├─ EDA.ipynb
│  ├─ EM_compare.ipynb
│  ├─ compare_json.ipynb
│  ├─ ensemble_inference.ipynb
│  ├─ korquad_json_to_datset.ipynb
│  ├─ nbest_analyze.ipynb
│  └─ readme.md
├─ postprocess.py
├─ retrieval.py # for comparing retrievers' performance
├─ train.py # for training the reader
├─ train_dpr.py # for training the dense retriever
├─ trainer
│  └─ DenseRetrievalTrainer.py
└─ utils
   ├─ arguments.py
   ├─ run_mlm.py
   ├─ trainer_qa.py
   └─ utils_qa.py

```
</div>
</details>

## 2️⃣ Team Introduction

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## 3️⃣ Data
![ODQA_data](https://user-images.githubusercontent.com/65378914/217733088-a82c1f7e-9739-4192-9e8c-7314fc4bcde0.png)

For the MRC (Machine Reading Comprehension) data, you can access it using the datasets library provided by HuggingFace. After storing the directory as dataset_name, you can load it as follows:
```
# To load the train_dataset
from datasets import load_from_disk
dataset = load_from_disk("./data/train_dataset/")
print(dataset)
```
The corpus used for retrieval, which is the set of documents, is stored in `./data/wikipedia_documents.json`. It consists of approximately 57,000 unique documents.
The dataset is stored in the pyarrow format for convenience using the datasets library. The following is the directory structure of `./data`:
```
# Entire dataset
./data/
    # Dataset used for training. Consists of train and validation sets.
    ./train_dataset/
    # Dataset used for submission. Consists of the validation set.
    ./test_dataset/
    # Wikipedia document corpus used for retrieval.
    ./wikipedia_documents.json
```
## Data Example
![ex](https://user-images.githubusercontent.com/65378914/217733295-1d6a3166-3582-454b-8e9b-01409b5e8597.png)

<details>
    <summary><b><font size="10">Label Description</font></b></summary>
<div markdown="1">

```
- id: Unique id of the question
- question: The question
- answers: Information about the answer. Each question has only one answer.
- answer_start : The starting position of the answer
- text: The text of the answer
- context: The document containing the answer
- title: The title of the document
- document_id: Unique id of the document
```
</div>
</details>

## 4️⃣ Model Description
## Reader
The Reader identifies potential sub-strings within the given context sentence that can serve as answers to the query sentence. <br>
The Reader utilizes the ModelForQuestionAnswering structure from the transformers library to compute the probabilities for each token in the context indicating the likelihood of it being the start or end point of the answer. <br>
The maximum length of the answer can be specified using `config.utils.max_answer_length`. <br>

## Retriever
The Retriever retrieves relevant documents from the database for the given query sentence. <br>
The number of documents retrieved can be specified using `config.retriever.topk`.
### 1. Sparse 
To use Sparse Embedding, select `sparse` as `config.path.type`.
#### (1) TF-IDF
Embeds the query sentence and context documents using Scikit-learn's TfidfVectorizer. The maximum size of the tf-idf vector can be specified using `config.retriever.sparse.tfidf_num_features`. After fitting, the tf-idf vectorized context documents and the TfidfVectorizer object will be stored in the `config.path.context` folder where the context sentences are stored.
#### (2) BM25

### 2. Dense
To use Dense Embedding, specify `dense` as `config.retriever.type`.

### Faiss
You can decide whether to use Faiss for retrieval by setting `config.retriever.faiss.use_faiss` to `True`. You can adjust the number of clusters created by IndexIVFScalarQuantizer using `config.retriever.faiss.num_clusters`, and the quantizer method used for indexing and distance calculation can be set using `config.retriever.faiss.metric`.

## 5️⃣ How to Run
## Environment Setup
```
## Test
```
python inference.py -c base_config
```
