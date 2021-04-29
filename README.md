# Horizontal-Federated-Learning-Demo

A horizontal FL image classification demo in book ***Practicing Federated Learning*** Chapter 3.

Performance comparison between FL model and centralized training model:
<div align="center">
    <img src="/figs/Accuracy.png" width="400"/><img src="/figs/Loss.png" width="400"/>
</div>

In order to obtain the centralized training result, just modify the following parameters in `config.py`:
- "no_models": 1
- "k": 1
- "local_epochs": 1

More details please refer to - [Practicing Federated Learning GitHub Repository](https://github.com/FederatedAI/Practicing-Federated-Learning)