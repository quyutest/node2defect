# node2defect
Supplementary code and data of the paper *Evaluating network embedding techniques in software bug prediction*

This repo can also be used as a complementary material for our previous paper:

Qu, Yu, Ting Liu, Jianlei Chi, Yangxu Jin, Di Cui, Ancheng He, and Qinghua Zheng. "node2defect: using network embedding to improve software defect prediction." In 2018 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 844-849. IEEE, 2018. https://dl.acm.org/doi/abs/10.1145/3238147.3240469

### Generating Class Dependency Network
---
In each subdirectory, we have already included the corresponding class dependency network (classgraph.dot). If you want to generate your own CDN, you can use the [Understand Perl Script](https://www.scitools.com/documents/manuals/pdf/understand_api.pdf) after installing the Understand tool, using the commend:
```bash
uperl Class-Graph.pl %YourOwnProjectDirectory%
```
### Generating the input file for network embedding algorithms
---
After generating the CDN, we can use the Driver.py to generate the input file for network embedding algorithms. After executing Driver.py, we can generate the "edgelist" file in each directory. For instance, we can use the [ProNE](https://github.com/THUDM/ProNE) implementation:
```bash
python proNE.py -graph edgelist -emb1 classgraph.emd -emb2 classgraph-2.emd -dimension 32 -step 10 -theta 0.5 -mu 0.2
```
### Run the experiment
---
After generating the classgraph.emd file, we can run the experiment by executing Node2Defect-Final-CrossValidation.py:
```bash
python Node2Defect-Final-CrossValidation.py
```
The overall results are listed in the files in the files like "All-Popt-%s-%s-%s.csv"


Requirements:  
python==2.7  
scipy==1.1.0  
networkx==1.7  
scikit-learn==0.19.1  
numpy==1.15.0  
pandas==0.22.0  
