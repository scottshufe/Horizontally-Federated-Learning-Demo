# Horizontally-Federated-Learning-Demo

A horizontally FL image classification demo in book ***Practicing Federated Learning*** Chapter 3.

Performance comparison between FL model and centralized training model:
<div align="left">
    <img src="/figs/Accuracy.png" width="350"/><img src="/figs/Loss.png" width="350"/>
</div>

In order to obtain the centralized training result, just modify the following parameters in `utils/config.json`:
- "no_models": 1
- "k": 1
- "local_epochs": 1

More details please refer to - [Practicing Federated Learning GitHub Repository](https://github.com/FederatedAI/Practicing-Federated-Learning)