# node2defect
Supplementary code and data of the paper *Evaluating network embedding techniques in software bug prediction*.

This repo can also be used as a complementary material for our previous paper:

Qu, Yu, Ting Liu, Jianlei Chi, Yangxu Jin, Di Cui, Ancheng He, and Qinghua Zheng. "node2defect: using network embedding to improve software defect prediction." In 2018 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 844-849. IEEE, 2018. https://dl.acm.org/doi/abs/10.1145/3238147.3240469

### 1. Generating Class Dependency Network
---
In each subdirectory, we have already included the corresponding Class Dependency Network (CDN) (classgraph.dot). If you want to generate your own CDN, you can use the [Understand Perl Script](https://www.scitools.com/documents/manuals/pdf/understand_api.pdf) file -- Class-Graph.pl, after installing the Understand tool, by using the commend:
```bash
uperl Class-Graph.pl %YourOwnProjectDirectory%
```
### 2. Generating the input file for network embedding algorithms
---
After generating the CDN, we can use the Driver.py to generate the input file for network embedding algorithms.
```bash
python Driver.py
```
After executing Driver.py, we can get the "edgelist" file in each directory. Then, for instance, we can use the [ProNE](https://github.com/THUDM/ProNE) implementation to generate the embedding file (classgraph.emd):
```bash
python proNE.py -graph edgelist -emb1 classgraph.emd -emb2 classgraph-2.emd -dimension 32 -step 10 -theta 0.5 -mu 0.2
```
### 3. Run the experiment
---
After generating the classgraph.emd file, we can run the experiment by executing Node2Defect-Final-CrossValidation.py:
```bash
python Node2Defect-Final-CrossValidation.py
```
The overall results are listed in the files in the files like "All-Popt-%s-%s-%s.csv"
### Requirements:  
python==3.7  
scipy==1.5.2  
networkx==2.5  
scikit-learn==0.23.2  
numpy==1.19.1  
pandas==1.1.2

We have used the following implementations of different network embedding algorithms, please refer to their repos for their dependencies:

[OpenNE](https://github.com/thunlp/OpenNE)  
[ProNE](https://github.com/THUDM/ProNE)  
[Walklets](https://github.com/benedekrozemberczki/walklets)  
