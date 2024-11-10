<h1 align="center">
    📖 Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception
</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2410.12788">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv">
    </a>
    <a href="https://huggingface.co/papers/2410.12788">
        <img alt="Hugging Face Daily Papers" src="https://img.shields.io/badge/Hugging_Face-Paper.svg?logo=huggingface">
    </a>
    <a href="https://opensource.org/license/apache-2-0">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-4285f4.svg?logo=apache">
    </a>
</p>

## CRUD
Original_Dataset/CRUD_RAG-main:
- *db_qa.txt* is the raw corpus related to question answering, filtered from *80000_docs*.
- *crud_split/split_merged.json* contains three types of question-answering datasets used for our evaluation.

Chunking_Result/CRUD_RAG-main:
- The *json* files store our chunking results.

## LongBench
Original_Dataset/LongBench-main:
- *data* contains raw corpora for the 8 question-answering datasets we used.

| Task                |   Task Type   | Eval metric | Avg len |    Language    | \#Sample |
| :------------------ | :-----------: | :---------: | :-----: | :------------: | :------: |
| HotpotQA            | Multi-doc QA  |     F1      |  9,151  |       EN       |   200    |
| 2WikiMultihopQA     | Multi-doc QA  |     F1      |  4,887  |       EN       |   200    |
| MuSiQue             | Multi-doc QA  |     F1      | 11,214  |       EN       |   200    |
| DuReader            | Multi-doc QA  |   Rouge-L   | 15,768  |       ZH       |   200    |
| MultiFieldQA-en     | Single-doc QA |     F1      |  4,559  |       EN       |   150    |
| MultiFieldQA-zh     | Single-doc QA |     F1      |  6,701  |       ZH       |   200    |
| NarrativeQA         | Single-doc QA |     F1      | 18,409  |       EN       |   200    |
| Qasper              | Single-doc QA |     F1      |  3,619  |       EN       |   200    |

Chunking_Result/LongBench-main:
- *a_chunk_ppl* contains chunking results using the PPL Chunking method.
- *b_chunk_prob_onlytwo* contains chunking results using the Margin Sampling Chunking method, where chunks are determined solely based on the preceding and following sentences.
- *c_chunk_prob* uses the preceding text chunk and the following sentence for Margin Sampling Chunking.
- *d_chunk_semantic* contains chunking results using semantic similarity.
- *LumberChunker_failure_log* contains some error logs that arise when other LLM chunking methods are difficult to apply to models of 7B or smaller.
- *tmp* contains the results of processing some raw datasets, mainly separating each document in the raw dataset for easier handling.

## MultiHop-RAG
Original_Dataset/MulithopQA-main:
- *data/corpus/corpus.txt* is the raw corpus document.
- *MultiHopRAG.json* stores the relevant data for question-answering evaluation.
- *tmp/corpus* stores each document in the raw corpus separately.

Chunking_Result/MulithopQA-main:
- Contains *ppl*, which are the chunking results using the PPL Chunking method.
- Contains *prob_onlytwo*, which are the chunking results using Margin Sampling Chunking based on the preceding and following sentences.
- Contains *prob*, which are the chunking results using Margin Sampling Chunking based on the preceding text chunk and the following sentence.
- Contains *semantic*, which are the chunking results using semantic similarity.

## RAGBench
Original_Dataset/RAGBench-main:
- *CUAD/test-00000-of-00001.parquet* is the raw corpus document.

Chunking_Result/RAGBench-main:
- *CUAD* stores our chunking results using the PPL Chunking method.


> Note: In order to avoid discrepancies caused by different tokenizers, we use the word count (using Python's split function) to calculate the average chunk length of English datasets, and use the character count to calculate the average chunk length of Chinese datasets.
