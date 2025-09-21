Deep Learning Assignments (IIT Hyderabad – AI2100/AI5100/EE6380)

This repository contains my solutions to the Deep Learning course assignments at IIT Hyderabad (Spring 2025).
Each assignment is implemented in a separate Jupyter Notebook (.ipynb) and covers key deep learning concepts — from foundational algorithms to modern architectures like Transformers and VAEs.

⸻

📂 Repository Structure
	•	Assignment 1 – Perceptron, Gradient Descent, and MLP
	•	Assignment 2 – 3D CNNs & Sequence Models (Elman, LSTM, GRU)
	•	Assignment 3 – Word2Vec & Vision Transformer (ViT)
	•	Assignment 4 – Autoencoders & Variational Autoencoders (VAE)

⸻

📘 Assignment Details

Assignment 1 – Fundamentals of Deep Learning
	•	Implemented Perceptron Learning Algorithm on linearly separable and noisy datasets.
	•	Implemented Gradient Descent for binary classification, comparing with Perceptron.
	•	Built an MLP (Multi-layer Perceptron) with a single hidden layer from scratch, including:
	•	Forward pass
	•	Backpropagation
	•	Gradient updates
	•	Experiments with weight initialization, learning rates, loss functions, and hidden layer size.

📓 Notebook: Assignment1.ipynb

⸻

Assignment 2 – Convolutional Neural Networks & Sequence Models
	•	Implemented a 3D Convolutional Neural Network (3D CNN) for MNIST point-cloud data:
	•	3D Convolution
	•	3D Pooling (Max, Avg, Global Avg)
	•	MLP classifier
	•	Complete 3D CNN pipeline with classification
	•	Designed a dataset for Balanced Parentheses Counting and trained sequence models:
	•	Elman RNN
	•	LSTM
	•	GRU
	•	Compared learning curves and performance against a simple baseline.

📓 Notebook: Assignment2.ipynb

⸻

Assignment 3 – Word2Vec & Vision Transformer
	•	Word2Vec (Skip-gram with Negative Sampling):
	•	Preprocessed text8 dataset (tokenization, vocabulary creation).
	•	Implemented Skip-gram with negative sampling from scratch.
	•	Trained embeddings and evaluated via:
	•	Visualization (SVD projections)
	•	Word similarity tasks (e.g., analogies like king - man + woman ≈ queen).
	•	Analyzed impact of hyperparameters.
	•	Vision Transformer (ViT) on CIFAR-10:
	•	Implemented an Encoder-only ViT: patch embeddings, positional encodings, Transformer encoder, CLS token.
	•	Trained and evaluated on CIFAR-10.
	•	Visualized attention maps across layers/heads to analyze spatial/contextual understanding.
	•	Experiments with hyperparameters (patch size, layers, heads).

📓 Notebook: Assignment3.ipynb

⸻

Assignment 4 – Autoencoders & Variational Autoencoders (VAE)
	•	Autoencoders on Fashion-MNIST:
	•	Latent dimensions: 16 and 48.
	•	Compared reconstruction errors and visual quality.
	•	Performed latent space interpolation to study smooth transitions between samples.
	•	Variational Autoencoder (VAE):
	•	12-dimensional latent space.
	•	Generated new Fashion-MNIST-like samples.
	•	Analyzed diversity & quality of generated outputs.
	•	Explored latent variable manipulation for controlled generation.

📓 Notebook: Assignment4.ipynb

⸻

🛠️ Technologies Used
	•	Python, NumPy, Matplotlib (core implementations from scratch)
	•	PyTorch (for tensor operations & autograd)
	•	Jupyter Notebook (experiments, analysis, and visualization)

⸻

