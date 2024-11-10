
<h1 align="center">
    Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception
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

**Meta-Chunking** leverages the capabilities of LLMs to flexibly partition documents into logically coherent, independent chunks. Our approach is grounded in a core principle: allowing variability in chunk size to more effectively capture and maintain the logical integrity of content. This dynamic adjustment of granularity ensures that each segmented chunk contains a complete and independent expression of ideas, thereby avoiding breaks in the logical chain during the segmentation process. This not only enhances the relevance of document retrieval but also improves content clarity.

> Note: Perplexity is a metric used to measure a language model's ability to predict text. It reflects the degree of uncertainty in generating the next token or sentence given a specific context. Our initial intuition was also to ensure that, during chunking, we split the text at points of certainty and keep it intact at points of uncertainty. This approach is more beneficial for subsequent retrieval and generation. Therefore, in fact, perplexity-based chunking leverages the hallucinations of language models to perceive text boundaries (relative to the boundaries of models), thereby ensuring that chunks are not split at points where language models hallucinate, avoiding the introduction of more hallucinations during retrieval and question answering by LLMs.

As illustrated in the following figure, example sentences exhibit a progressive relationship, yet their semantic similarity is low, which may result in their complete separation.

![Comparison Figure](images/figure1.jpg)

## Highlights

- Introduces the concept of Meta-Chunking, which operates at a granularity between sentences and paragraphs.

- Propose two implementation strategies: Margin Sampling Chunking and Perplexity (PPL) Chunking.

  ![Framework](images/figure2.png)

- Put forward a Meta-Chunking with dynamic combination strategy designed to achieve a valid balance between fine-grained and coarse-grained text segmentation.

- Extensive experiments were conducted on eleven datasets across four benchmark.

## Quick Start

```
# Install dependencies
conda create -n MetaChunking python=3.10
conda activate MetaChunking
pip install -r requirements.txt

# Run the demo
python app.py
```
The four benchmarks used in this paper are as follows, and you can find the relevant datasets and evaluation methods through the links: [CRUD](https://github.com/IAAR-Shanghai/CRUD_RAG)，[LongBench](https://github.com/THUDM/LongBench)，[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG)，[RAGBench](https://github.com/rudaoshi/RAG-Bench). Additionally, for quick and easy use, we provide you with the datasets and chunking results, which can be downloaded via [Google Drive](https://drive.google.com/file/d/1nUPV6hSOZHhlakmlDFPpdBCmLjI5tB_a/view?usp=drive_link). For specific configurations of chunking and evaluation for each benchmark, please refer to [Instructions.md](https://github.com/IAAR-Shanghai/Meta-Chunking/blob/main/Instructions.md).

## Results

![Main result](images/figure3.png)

![Two PPL Chunking strategies](images/figure4.jpg)

# Notes

- We conducted extensive experiments on four benchmarks. Since each benchmark has many parameters to set, for the reproducibility of the experiment, we set up an independent folder for each benchmark. The datasets of benchmarks and their usage can be found on GitHub.
- The **meta_chunking** folder contains chunking programs for the four benchmarks, which share the same principles and implementation methods. The **eval** folder includes evaluation methods for the four benchmarks. For a benchmark, we first divide the dataset into appropriate chunks, establish a vector database, generate answers to questions, and then evaluate the impact of chunking on relevant metrics.
- We provide a Gradio chunking program in the **example** folder, which can be operated by running app.py. You can also dynamically adjust the parameters according to your chunking needs.


## Citation

```
@article{MetaChunking,
  title={Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception},
  author={Zhao, Jihao and Ji, Zhiyuan and Qi, Pengnian and Niu, Simin and Tang, Bo and Xiong, Feiyu and Li, Zhiyu},
  journal={arXiv preprint arXiv:2410.12788},
  year={2024}
}
```

