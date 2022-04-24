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
@misc{https://doi.org/10.48550/arxiv.2204.03905,
  doi = {10.48550/ARXIV.2204.03905},
  url = {https://arxiv.org/abs/2204.03905},
  author = {Yuan, Hongyi and Yuan, Zheng and Gan, Ruyi and Zhang, Jiaxing and Xie, Yutao and Yu, Sheng},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model},
  publisher = {arXiv},
  year = {2022}
}
```
