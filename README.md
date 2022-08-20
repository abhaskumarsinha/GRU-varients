# GRU-varients
An implementation of classical GRU (Cho, el at. 2014) along with Optimized versions (Dey, Rahul. 2017) on TensorFlow that outperforms Native tf.keras.layers.GRU(units) implementation of Keras.

## Dataset
The original paper [Dey, Rahul, et al. 2017] used MNIST Dataset for the training and inference purpose, we'll be comparing them on `np.random.rand(batch_size, time_width, 500)` in our implementation (this ensures that bias ~ 0, following random noise cancel each others.) 

## Abstract
The whole code has two files - `CustomGRU.py` and `Notebook.ipynb` as different implementations of Gated Recurrent Units and an short tutorial on usage in form of Python Notebooks respectively. We compare GRU0 - Classical GRU, GRU1/GRU2/GRU3 as optimized GRU models and GRU4 as Native TensorFlow implementation of GRU in form of `tf.keras.layers.GRU(units)`. GRUs are special RNNs which are very lighter alternative to LSTM with very few parameters and almost similar performance [Cho, et al. 2014]. We optimize classical GRUs to the versions in [Dey, Rahul. 2017] for our analysis. We found that all four manual implementations outperform the Native TensorFlow GRU implementation, which almost stopped converging after a certain epochs. Our models had slower initial convergence compared to TensorFlow's native implementation which was much sharper in the beginning but totally stopped after few epochs whereas our Models continued it. Training time was almost same for all the five models, though, GRU2/GRU3 had much faster training time and GRU4 outperformed all in training duration.

## Results

### Published Results vs Our finding.

![Screenshot 2022-08-20 020700](https://user-images.githubusercontent.com/31654395/185734150-d166d680-8f5b-43da-8568-32f83770fc4e.png)
(Trained on MNIST Dataset)

![download (2)](https://user-images.githubusercontent.com/31654395/185734162-0c511d65-0f30-4ffa-829f-e171a721d54d.png)
(Same Models, trained on Random Noise)

### Comparision with Native TensorFlow Keras Model

![download (3)](https://user-images.githubusercontent.com/31654395/185734193-d7ab442d-40da-4e7c-8c4d-b213acfd8ff7.png)
(Blue line refers to TensorFlow Keras Native GRU Implementation)

## Equations of update/reset gate in Optimized Models.

![Screenshot 2022-08-20 022808](https://user-images.githubusercontent.com/31654395/185734525-d741a886-8bc6-496d-89d4-1cb6d987a6e2.png)


## References

1. Dey, Rahul, and Fathi M. Salem. "Gate-variants of gated recurrent unit (GRU) neural networks." 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS). IEEE, 2017.

2. Bengio, Y., Simard, P., and Frasconi, P. Learning Longterm Dependencies with Gradient Descent is Difficult. IEEE Trans.Neural Networks, 5(2):157–166, 1994. H. Simpson, Dumb Robots, 3rd ed., Springfield: UOS Press, 2004, pp.6-9. 

3. Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).

4. Bengio, Yoshua, Patrice Simard, and Paolo Frasconi. "Learning long-term dependencies with gradient descent is difficult." IEEE transactions on neural networks 5.2 (1994): 157-166.

5. Bengio, Yoshua, Nicolas Boulanger-Lewandowski, and Razvan Pascanu. "Advances in optimizing recurrent networks." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013.

6. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.

