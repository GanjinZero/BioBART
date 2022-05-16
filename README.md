# BioBART
BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model [ACL-BioNLP 2022] [Paper](https://arxiv.org/abs/2204.03905)

Tsinghua University \& International Digital Economy Academy.

# Model Checkpoints

- Base Version (6 + 6 Layers): **GanjinZero/biobart-base** or **IDEA-CCNL/Yuyuan-Bart-139M** (same model)
- Large Version (12 + 12 Layers): **GanjinZero/biobart-large** or **IDEA-CCNL/Yuyuan-Bart-400M** (same model)

P.S. Yuyuan is a character in novel Fengshenbang. [Chinese Introduction](https://baike.baidu.com/item/%E9%A4%98%E5%85%83/968026) \ [English Introduction](https://en.wikisource.org/wiki/Portal:Investiture_of_the_Gods/Chapter_75)

Two line usages:
```python
model = AutoModel.from_pretrained('GanjinZero/biobart-base')
# model = AutoModel.from_pretrained('GanjinZero/biobart-large')
tok = AutoTokenizer.from_pretrained('GanjinZero/biobart-base')
```

# Citation
```bibtex
@inproceedings{yuan-etal-2022-biobart,
    title = "{B}io{BART}: Pretraining and Evaluation of A Biomedical Generative Language Model",
    author = "Yuan, Hongyi  and
      Yuan, Zheng  and
      Gan, Ruyi  and
      Zhang, Jiaxing  and
      Xie, Yutao  and
      Yu, Sheng",
    booktitle = "Proceedings of the 21st Workshop on Biomedical Language Processing",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.bionlp-1.9",
    pages = "97--109",
    abstract = "Pretrained language models have served as important backbones for natural language processing. Recently, in-domain pretraining has been shown to benefit various domain-specific downstream tasks. In the biomedical domain, natural language generation (NLG) tasks are of critical importance, while understudied. Approaching natural language understanding (NLU) tasks as NLG achieves satisfying performance in the general domain through constrained language generation or language prompting. We emphasize the lack of in-domain generative language models and the unsystematic generative downstream benchmarks in the biomedical domain, hindering the development of the research community. In this work, we introduce the generative language model BioBART that adapts BART to the biomedical domain. We collate various biomedical language generation tasks including dialogue, summarization, entity linking, and named entity recognition. BioBART pretrained on PubMed abstracts has enhanced performance compared to BART and set strong baselines on several tasks. Furthermore, we conduct ablation studies on the pretraining tasks for BioBART and find that sentence permutation has negative effects on downstream tasks.",
}
```
