# Machine Learning Labs — NTHU CS

Six Jupyter labs from the NTHU CS Machine Learning course. They build classic models and neural networks from scratch in NumPy: regression, decision trees, linear classifiers, an MLP, a CNN and an RNN.

## Assignments

Results are the metrics printed in each submitted notebook, on a held-out validation split unless noted. The labs also produced test-set predictions for class Kaggle competitions.

| # | Topic | Dataset | Methods | Result | Notebook |
|---|-------|---------|---------|--------|----------|
| 1 | Regression | Grip-force data. Basic part: weight → grip force. Advanced part: age, gender, height, weight, body fat and blood pressure → grip force | Gradient-descent linear and polynomial regression in NumPy. IQR outlier removal. The advanced part trains separate multivariate models for men and women. | MAPE 17.67% (linear), 17.50% (degree-4 polynomial). Advanced: MAPE 10.71% (male) and 11.38% (female) | [Lab1.ipynb](ML_HW1/Lab1_submit/Lab1.ipynb) |
| 2 | Decision tree & random forest | ICU patient records (APACHE vitals and labs, comorbidities) → in-hospital death | Decision tree from scratch (entropy, information gain, best-split search). Random forest with 100 trees, bootstrap sampling, random feature subsets and majority vote. | F1 0.7422 (random forest) | [basic](ML_HW2/Lab2_submit/lab2_basic.ipynb), [advanced](ML_HW2/Lab2_submit/lab2_advance.ipynb) |
| 3 | Linear classifiers | Age, BMI and glucose → diabetes (25k training records) | Perceptron, Fisher's linear discriminant (LDA), and LDA with Gaussian class-conditional densities and MAP decision, all from scratch | Accuracy 53.92% / F1 0.30 (Perceptron). 68.88% / F1 0.68 (LDA). 68.82% / F1 0.69 (LDA + MAP) | [Lab3.ipynb](ML_HW3/Lab3_submit/Lab3.ipynb) |
| 4 | Deep neural network (MLP) | Synthetic `sin(2·sin(2·sin(2·sin x)))` curve. Retinal OCT images (28×28): CNV vs. normal, then 4 classes (CNV, DME, drusen, normal) | NumPy MLP framework: dense layers, sigmoid/ReLU/softmax, BCE/CCE/MSE losses, backpropagation, mini-batch SGD with learning-rate decay | Binary: 92.58% accuracy, macro-F1 0.93. 4-class: 84.96% accuracy, macro-F1 0.84. Regression: final training MSE about 6.5e-4 | [Lab4.ipynb](ML_HW4/Lab4_submit/Lab4.ipynb) |
| 5 | Convolutional neural network | Chest X-rays (32×32, 600 training images): normal vs. abnormal | NumPy CNN: convolution layer (zero padding, forward and backward), max pooling with masks, flatten. Model is Conv → ReLU → MaxPool → Dense → Sigmoid. | Layer outputs checked against reference values. No validation metric printed in the notebook | [Lab5.ipynb](ML_HW5/Lab5_submit/Lab5.ipynb) |
| 6 | Recurrent neural network | Synthetic sine-wave sequences (predict the next value). Accelerometer and gyroscope time series (128 steps × 6 channels) → 3 activity classes | NumPy RNN layer with backpropagation through time. Compares a dense-only model with an RNN model. The activity model is RNN(256) → Dense → Softmax. | Sine wave validation MSE: 0.140 (dense only), 0.134 (RNN). Activity validation CCE loss 0.356 | [Lab6.ipynb](ML_HW6/Lab6_submit/Lab6.ipynb) |

Each `ML_HW*` folder contains:

- `Lab*_submit/`: my notebook, prediction outputs and report.
- `Lab*_template/`: the course-provided starter notebook.
- `Spec.pdf` (`spec.pdf` in HW1): the assignment specification.
- `TestingData/` (`TestingDate/` in HW2): the provided datasets.

## Tech stack

- Python 3, Jupyter Notebook, Google Colab
- NumPy for all model code: forward and backward passes, gradient descent, tree building
- pandas for data loading and cleaning. Matplotlib for plots and training animations.
- scikit-learn only for helpers: F1 score, train/validation split, scaling and one-hot encoding

## Final project

**[Handwritten Mathematical Expression Recognition](https://github.com/ivylingchiang/handwritten-math-recognition)** converts images of handwritten formulas to LaTeX. It uses a CNN encoder and an attention-based RNN decoder.
