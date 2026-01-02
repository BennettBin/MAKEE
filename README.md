# MAKEE: Multi-view Attribute Network and Sequence Embedding Approach for Predictive Process Monitoring



## About MAKEE

Predictive business process monitoring can help detect and solve problems on time by monitoring the execution of business processes in real time, thereby improving overall business efficiency and performance. Current deep learning-based studies have found that embedding structural information of process models helps neural networks learn the deep logic behind business processes. However, they mainly focus on the control-flow perspective, while other perspectives behind the business process, such as organizational structure, social network, and resource behavior, have been largely overlooked. To address this issue, this study proposes a multi-view learning prediction approach that integrates complementary information from both multiple attribute networks and sequences. We carefully design a deep learning model framework to integrate multi-view structural and sequential information for the next-activity prediction of the running trace. On the one hand, a simple and efficient process mining algorithm is designed to model multiple attribute network graphs, and a graph convolutional network is integrated to learn their multi-view structural information, helping understand the deep features of business scenarios. For this, a node feature enhancement method is proposed to integrate global information from historical business executions to help the proposed neural network understand the structure of a complete business scenario. On the other hand, we construct the feature representation of attribute sequences and integrate the Transformer to capture the dependency relations and sequential features within attribute sequences. Experimental evaluation of twelve real-life event logs shows that the proposed approach performs well in prediction accuracy and robustness.



## Learning Laravel

Laravel has the most extensive and thorough documentation and video tutorial library of all modern web application frameworks, making it a breeze to get started with the framework.

If you don't feel like reading, [Laracasts](https://laracasts.com) can help.  
Laracasts contains over 1500 video tutorials on a range of topics including Laravel, modern PHP, unit testing, and JavaScript.

## Laravel Sponsors

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
