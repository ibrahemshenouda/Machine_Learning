# Autoencoder Labs

This folder contains implementations of autoencoder network architectures for dimensionality reduction, image denoising, and anomaly detection.

## Notebooks
* **`Autoencoder_MNIST.ipynb`**: Implementation of a basic dense autoencoder (784 -> 32 -> 784 bottleneck representation) on MNIST digits dataset.
* **`Denoising_Autoencoder_MNIST.ipynb`**: Autoencoder trained with synthetic gaussian noise injection to reconstruct clean MNIST digit images.
* **`Autoencoder_Anomaly_Detection_CreditCard.ipynb`**: Reconstruction-based anomaly detector trained exclusively on normal transactions to identify credit card fraud by flagging high reconstruction error samples.
