# MAKEE: Multi-view Attribute Network and Sequence Embedding Approach for Predictive Process Monitoring

## Abstract

Predictive business process monitoring can help detect and solve problems on time by monitoring the execution of business processes in real time, thereby improving overall business efficiency and performance. Current deep learning-based studies have found that embedding structural information of process models helps neural networks learn the deep logic behind business processes. However, they mainly focus on the control-flow perspective, while other perspectives behind the business process, such as organizational structure, social network, and resource behavior, have been largely overlooked. To address this issue, this study proposes a multi-view learning prediction approach that integrates complementary information from both multiple attribute networks and sequences. We carefully design a deep learning model framework to integrate multi-view structural and sequential information for the next-activity prediction of the running trace. On the one hand, a simple and efficient process mining algorithm is designed to model multiple attribute network graphs, and a graph convolutional network is integrated to learn their multi-view structural information, helping understand the deep features of business scenarios. For this, a node feature enhancement method is proposed to integrate global information from historical business executions to help the proposed neural network understand the structure of a complete business scenario. On the other hand, we construct the feature representation of attribute sequences and integrate the Transformer to capture the dependency relations and sequential features within attribute sequences. Experimental evaluation of twelve real-life event logs shows that the proposed approach performs well in prediction accuracy and robustness.

## Learning Laravel

Laravel has the most extensive and thorough documentation and video tutorial library of all modern web application frameworks, making it a breeze to get started with the framework.

If you don't feel like reading, [Laracasts](https://laracasts.com) can help.  
Laracasts contains over 1500 video tutorials on a range of topics including Laravel, modern PHP, unit testing, and JavaScript.

## Evaluation
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|Metrics | Approaches | BPIC 2012W | BPIC 2012 WC | BPIC 2020 D | BPIC 2020 I | BPIC 2020 Pe | BPIC2020 Pr | BPIC 2020 R | BPIC 2013 P | BPIC 2019 W | BPIC 2019 C | Env permit | Nasa |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|(1) | 0.8997 | 0.7982 | 0.8838 | 0.8733 | 0.8056 | 0.8485 | 0.8910 | 0.6027 | 0.7964 | 0.7512 | 0.8249 | 0.8811 |
|(2) | 0.8988 | 0.7886 | 0.8603 | 0.8678 | 0.8155 | 0.8581 | 0.8579 | 0.5806 | 0.8616 | 0.8507 | 0.8235 | - |
|(3) | 0.9053 | 0.7857 | 0.8323 | 0.8267 | 0.7767 | 0.7973 | 0.8313 | 0.5277 | 0.8333 | 0.8493 | 0.7807 | 0.8760 |
|(4)-B | 0.6266 | 0.5445 | 0.5407 | 0.4576 | 0.3273 | 0.3483 | - | 0.5003 | 0.5554 | 0.8354 | - | 0.8084 |
|(4)-LB | 0.6749 | 0.5807 | 0.6362 | 0.5079 | 0.4862 | 0.6336 | - | 0.5235 | 0.6038 | 0.8330 | - | 0.8331 |
|(4)-W | 0.6210 | 0.5448 | 0.5431 | 0.4875 | 0.4350 | 0.4833 | - | 0.5119 | 0.5918 | 0.8276 | - | 0.8427 |
|(4)-LW | 0.6920 | 0.5909 | 0.6870 | 0.5455 | 0.4598 | 0.5780 | - | 0.4910 | 0.6092 | 0.8320 | - | 0.8413 |
|(5) | 0.8927 | 0.7718 | 0.8930 | 0.8734 | 0.8180 | 0.8556 | 0.8830 | 0.5064 | 0.8309 | 0.8450 | 0.7913 | 0.8747 |
|MAKEE | **0.9104** | **0.8555** | **0.9085** | **0.8835** | **0.8232** | **0.8627** | **0.9025** | **0.6389** | **0.8823** | **0.8740** | **0.8329** | **0.8872** |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|(1) | 0.7566 | 0.6350 | 0.4648 | 0.4808 | 0.4089 | 0.4720 | 0.4367 | 0.3956 | 0.2970 | 0.0780 | 0.3279 | 0.8182 |
|(2) | 0.7538 | 0.6255 | 0.4229 | 0.4550 | 0.3496 | 0.4236 | 0.4133 | 0.2862 | 0.3701 | 0.1891 | 0.2035 | - |
|(3) | **0.7973** | 0.6510 | 0.4633 | 0.4487 | 0.4193 | 0.4413 | 0.4320 | 0.2893 | 0.3890 | 0.3090 | 0.3973 | 0.8203 |
|(4)-B | 0.3298 | 0.2858 | 0.2081 | 0.1703 | 0.0970 | 0.1195 | - | 0.2703 | 0.1704 | 0.2082 | - | 0.0186 |
|(4)-LB | 0.4547 | 0.3187 | 0.2510 | 0.2018 | 0.1744 | 0.2870 | - | 0.2891 | 0.2045 | 0.1891 | - | 0.0189 |
|(4)-W | 0.3704 | 0.2950 | 0.2081 | 0.1747 | 0.1166 | 0.1432 | - | 0.2920 | 0.2159 | 0.1779 | - | 0.0191 |
|(4)-LW | 0.4746 | 0.3840 | 0.2844 | 0.2396 | 0.1540 | 0.2464 | - | 0.2440 | 0.2137 | 0.2067 | - | 0.0190 |
|(5)-RGB | 0.7594 | 0.6090 | **0.5738** | **0.5091** | 0.4144 | 0.5111 | 0.4485 | 0.2442 | 0.4743 | 0.2141 | 0.4014 | 0.8216 |
|MAKEE | 0.7630 | **0.7298** | 0.5628 | 0.4694 | **0.4234** | **0.5136** | **0.5582** | 0.3445 | **0.4969** | **0.5088** | **0.4091** | **0.8249** |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|


We would like to extend our thanks to the following sponsors for funding Laravel development.  
If you are interested in becoming a sponsor, please visit the Laravel sponsorship page.

### Premium Partners

- **Vehikl**
- **Tighten Co.**
- **WebReinvent**
- **Kirschbaum Development Group**
- **64 Robots**
- **Cubet Techno Labs**
- **Cyber-Duck**
- **Many**
- **Webdock**
- **DevSquad**
- **Jump24**
- **Redberry**
- **Active Logic**
- **byte5**
- **OP.GG**

## Contributing

Thank you for considering contributing to the Laravel framework!  
The contribution guide can be found in the Laravel documentation.

## Code of Conduct

In order to ensure that the Laravel community is welcoming to all,  
please review and abide by the Code of Conduct.

## Security Vulnerabilities

If you discover a security vulnerability within Laravel,  
please send an e-mail to Taylor Otwell via taylor@laravel.com.  
All security vulnerabilities will be promptly addressed.

## License

The Laravel framework is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).
