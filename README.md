# ot_generative_models
Implementation of generative models using optimal transport, for the spring 2021 class of "Optimal Transport: theory, computations, statistics and ML applications". Final grade: 20/20.

**Sinkhorn AutoDiff:**
*"Learning Generative Models with Sinkhorn Divergences"* by Aude Genevay, Gabriel Peyré, Marco Cuturi (Link: https://arxiv.org/abs/1706.00292)

**OT-GAN**
*"Improving GANs Using Optimal Transport"* by Tim Salimans, Han Zhang, Alec Radford, Dimitris Metaxas (Link: https://arxiv.org/abs/1803.05573)

# Project structure

* `main.py`: includes the `main` function used to train a model, save and store results.
* `architectures.py`: includes the achitectures used by the models (classes `Generator`, `Critic`, `ConvGenerator`, `ConvCritic`).
* `data_preprocessing.py`: includes `mnist_transforms` to pre-process MNIST data. 
* `simulated_data.py`: includes the `GaussianToy` class used to build the 2D Gaussian Mixture dataset.
* `utils.py`: includes various useful functions (functions `plot_grid`, `generate_plot_grid`, `pairwise_cosine_distance`, `sinkhorn_divergence` class `CReLU`).
* `models/sinhorn_gan.py`: includes the loss function `sinhorn_loss` and the training function `train_sinkhorn_gan`.
* `models/ot_gan.py`: includes the loss function `minibatch_energy_distance` and the training function `train_ot_gan`.

# Instructions

File `main.py` defines an argument parser, with all arguments needed to train the models, along with a short description for each of them.

To train a model, use:
```
cd ot_generative_models
python python main.py --args
```

Examples are provided in the following notebook: https://colab.research.google.com/drive/1gxYrXVwTAwIbAR1FX8W1I2l7Uileozdq#scrollTo=E0KNaHFo6WPk.
