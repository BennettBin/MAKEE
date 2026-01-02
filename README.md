# MAKEE: Multi-view Attribute Network and Sequence Embedding Approach for Predictive Process Monitoring

## Problem introduction
The implementation process of real-time PBPM usually involves three stages:  
(1) The PAIS records business activities and their related attributes during business operations in real-time, such as operators, time, equipment, and resources, and then stores the recorded information in a structured manner in event logs.  
(2) The recorded event logs and other business process information become training inputs for the PBPM algorithm after preprocessing. The predictive algorithm can acquire predictive ability by learning the implicit business rules or structures involved.  
(3) Depending on the characteristics of the PAIS, business process managers deploy the learned predictive model obtained in the information system. The PAIS captures new events in real time and feeds the current prefix sequence into the predictive model, thereby achieving real-time PBPM for process managers. 

## About MAKEE
Current deep learning-based studies have found that embedding structural information of process models helps neural networks learn the deep logic behind business processes. However, they mainly focus on the control-flow perspective, while other perspectives behind the business process, such as organizational structure, social network, and resource behavior, have been largely overlooked. To address this issue, this study proposes a multi-view learning prediction approach that integrates complementary information from both multiple attribute networks and sequences. We carefully design a deep learning model framework to integrate multi-view structural and sequential information for the next-activity prediction of the running trace. On the one hand, a simple and efficient process mining algorithm is designed to model multiple attribute network graphs, and a graph convolutional network is integrated to learn their multi-view structural information, helping understand the deep features of business scenarios. For this, a node feature enhancement method is proposed to integrate global information from historical business executions to help the proposed neural network understand the structure of a complete business scenario. On the other hand, we construct the feature representation of attribute sequences and integrate the Transformer to capture the dependency relations and sequential features within attribute sequences. 

## Evaluation
To verify the effectiveness and progressiveness of MAKEE, we have selected several advanced methods as benchmarks. The methods under consideration provide source code for reproduction, which enables us to conduct credible comparisons under the same experimental settings in the source papers. Specifically, the following methods were selected for comparison:  
(1) Predictive approach combining one-hot encoding and LSTM to predict timestamps and activities \cite{tax2017predictive};  
(2) Predictive approach based on inception CNN model \cite{di2019activity};  
(3) Multi-view predictive approach based on LSTM \cite{pasquadibisceglie2021multi};  
(4) Predictive approach based on GCN model with four variants, i.e., ${GCN}_B$, ${GCN}_{LB}$, ${GCN}_W$, and ${GCN}_{LW}$ \cite{venugopal2021comparison};  
(5) Predictive approach combining RGB encoding and CNN to predict activities \cite{pasquadibisceglie2020predictive}.  

| Metrics | Approaches | BPIC 2012W | BPIC 2012 WC | BPIC 2020 D | BPIC 2020 I | BPIC 2020 Pe | BPIC2020 Pr | BPIC 2020 R | BPIC 2013 P | BPIC 2019 W | BPIC 2019 C | Env permit | Nasa |
|----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Accuracy | (1) | 0.8997 | 0.7982 | 0.8838 | 0.8733 | 0.8056 | 0.8485 | 0.8910 | 0.6027 | 0.7964 | 0.7512 | 0.8249 | 0.8811 |
|          | (2) | 0.8988 | 0.7886 | 0.8603 | 0.8678 | 0.8155 | 0.8581 | 0.8579 | 0.5806 | 0.8616 | 0.8507 | 0.8235 | - |
|          | (3) | 0.9053 | 0.7857 | 0.8323 | 0.8267 | 0.7767 | 0.7973 | 0.8313 | 0.5277 | 0.8333 | 0.8493 | 0.7807 | 0.8760 |
|          | (4)-B | 0.6266 | 0.5445 | 0.5407 | 0.4576 | 0.3273 | 0.3483 | - | 0.5003 | 0.5554 | 0.8354 | - | 0.8084 |
|          | (4)-LB | 0.6749 | 0.5807 | 0.6362 | 0.5079 | 0.4862 | 0.6336 | - | 0.5235 | 0.6038 | 0.8330 | - | 0.8331 |
|          | (4)-W | 0.6210 | 0.5448 | 0.5431 | 0.4875 | 0.4350 | 0.4833 | - | 0.5119 | 0.5918 | 0.8276 | - | 0.8427 |
|          | (4)-LW | 0.6920 | 0.5909 | 0.6870 | 0.5455 | 0.4598 | 0.5780 | - | 0.4910 | 0.6092 | 0.8320 | - | 0.8413 |
|          | (5) | 0.8927 | 0.7718 | 0.8930 | 0.8734 | 0.8180 | 0.8556 | 0.8830 | 0.5064 | 0.8309 | 0.8450 | 0.7913 | 0.8747 |
|          | MAKEE | 0.9104 | 0.8555 | 0.9085 | 0.8835 | 0.8232 | 0.8627 | 0.9025 | 0.6389 | 0.8823 | 0.8740 | 0.8329 | 0.8872 |




## Contributing
(1) We propose an approach to simultaneously learn explicit sequential information from attribute sequences and deep structural information from event attribute networks, providing a novel perspective for PBPM research.  
(2) We adopt the idea of multi-view learning, treating each event attribute as a separate view. Our carefully designed neural network model effectively integrates multiple structural features from attribute networks and multiple sequence features from prefix sequences, enabling the model to recognize differentiated states of events from various perspectives.  
(3) The node feature enhancement method has been proposed to integrate the global structural information of attribute networks into their feature encoding, helping the neural network to have a more comprehensive understanding of a complete business scenario.  
(4) We compare our method with benchmark methods based on twelve real-life event logs and conduct multiple evaluations from different perspectives to explore its effectiveness and robustness for the next-activity prediction task.  

## Reference
If you use this method, please cite the original paper:  
Chen, B., Zhao, S., Lin, L., & Zhang, Q. (2025). MAKEE: Multi-view Attribute Network and Sequence Embedding Approach for Predictive Process Monitoring. Knowledge-Based Systems, 114299.
