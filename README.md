
<h1 align="center">
    Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception
</h1>
<p align="center">
    <a href="">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv">
    </a>
    <a href="https://opensource.org/license/apache-2-0">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-4285f4.svg?logo=apache">
    </a>
</p>

**Meta-Chunking** leverages the capabilities of LLMs to flexibly partition documents into logically coherent, independent chunks. Our approach is grounded in a core principle: allowing variability in chunk size to more effectively capture and maintain the logical integrity of content. This dynamic adjustment of granularity ensures that each segmented chunk contains a complete and independent expression of ideas, thereby avoiding breaks in the logical chain during the segmentation process. This not only enhances the relevance of document retrieval but also improves content clarity.

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

## Results

![Main result](images/figure3.png)

![Two PPL Chunking strategies](images/figure4.jpg)

# Notes

- We conducted extensive experiments on four benchmarks. Since each benchmark has many parameters to set, for the reproducibility of the experiment, we set up an independent folder for each benchmark. The datasets of benchmarks and their usage can be found on GitHub.
- The **meta_chunking** folder contains chunking programs for the four benchmarks, which share the same principles and implementation methods. The **eval** folder includes evaluation methods for the four benchmarks. For a benchmark, we first divide the dataset into appropriate chunks, establish a vector database, generate answers to questions, and then evaluate the impact of chunking on relevant metrics.
- We provide a Gradio chunking program in the **example** folder, which can be operated by running app.py. You can also dynamically adjust the parameters according to your chunking needs.


## Citation

```

```

