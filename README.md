````md
# ML
# Heirarchy

ARTIFICIAL INTELLIGENCE (AI)
│
├── Machine Learning (ML)
│     │
│     ├── Supervised Learning
│     │       ├── Classification
│     │       │       ├── Problem definition & use-cases
│     │       │       ├── Algorithms:
│     │       │       │       ├── Logistic Regression
│     │       │       │       ├── k-Nearest Neighbors (kNN)
│     │       │       │       ├── Decision Trees
│     │       │       │       ├── Random Forest
│     │       │       │       ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│     │       │       │       ├── Support Vector Machines (SVM)
│     │       │       │       └── Neural Networks (MLP, CNN for images)
│     │       │       ├── Preprocessing:
│     │       │       │       ├── Handling class imbalance (SMOTE, class weights)
│     │       │       │       ├── Encoding categorical features (One-hot, Target encoding)
│     │       │       │       └── Feature scaling (StandardScaler, MinMax)
│     │       │       ├── Feature engineering:
│     │       │       │       ├── Interaction terms
│     │       │       │       ├── Binning, polynomial features
│     │       │       │       └── Embeddings for high-cardinality categorical data
│     │       │       ├── Evaluation metrics:
│     │       │       │       ├── Accuracy (when balanced)
│     │       │       │       ├── Precision / Recall / F1-score
│     │       │       │       ├── ROC-AUC, PR-AUC
│     │       │       │       └── Confusion matrix, calibration
│     │       │       ├── Cross-validation:
│     │       │       │       ├── k-fold, stratified k-fold
│     │       │       │       └── Time-series CV (rolling window)
│     │       │       └── Practical projects:
│     │       │               ├── Spam detector
│     │       │               ├── Medical diagnosis (binary/multi-class)
│     │       │               └── Multi-class image classifier (Cats vs Dogs vs Others)
│     │       │
│     │       ├── Regression
│     │       │       ├── Problem definition & use-cases
│     │       │       ├── Algorithms:
│     │       │       │       ├── Linear Regression / OLS
│     │       │       │       ├── Ridge / Lasso (regularized regression)
│     │       │       │       ├── Decision Tree Regressor
│     │       │       │       ├── Random Forest Regressor
│     │       │       │       ├── Gradient Boosting Regressors (XGBoost, LightGBM)
│     │       │       │       └── Neural Networks for regression
│     │       │       ├── Preprocessing:
│     │       │       │       ├── Handling outliers
│     │       │       │       ├── Feature scaling/normalization
│     │       │       │       └── Transformations (log, Box-Cox)
│     │       │       ├── Evaluation metrics:
│     │       │       │       ├── Mean Absolute Error (MAE)
│     │       │       │       ├── Mean Squared Error (MSE) / RMSE
│     │       │       │       ├── R² (coefficient of determination)
│     │       │       │       └── Mean Absolute Percentage Error (MAPE)
│     │       │       └── Practical projects:
│     │       │               ├── House price prediction
│     │       │               ├── Demand forecasting (short-term)
│     │       │               └── Energy consumption prediction
│     │       │
│     │       ├── Common supervised skills
│     │       │       ├── Bias-variance tradeoff
│     │       │       ├── Regularization and hyperparameter tuning (GridSearchCV, RandomSearch, Bayesian)
│     │       │       ├── Model interpretability (SHAP, LIME, feature importances)
│     │       │       └── Pipeline building with scikit-learn
│     │
│     ├── Unsupervised Learning
│     │       ├── Clustering
│     │       │       ├── Algorithms: K-Means, Hierarchical, DBSCAN, MeanShift, Gaussian Mixture Models
│     │       │       ├── Preprocessing: scaling, PCA/TSNE for visualization
│     │       │       ├── Evaluation (silhouette score, Davies-Bouldin)
│     │       │       └── Projects: customer segmentation, anomaly detection
│     │       ├── Dimensionality Reduction
│     │       │       ├── PCA (theory & kernel PCA)
│     │       │       ├── t-SNE, UMAP (visualization)
│     │       │       └── Use-cases: noise reduction, visualization, speeding up models
│     │       └── Association Rule Mining
│     │               ├── Apriori, FP-Growth
│     │               └── Market basket analysis
│     │
│     ├── Semi-Supervised Learning
│     │       ├── Self-training, pseudo-labeling
│     │       ├── Graph-based label propagation
│     │       └── When labeled data is scarce
│     │
│     └── Reinforcement Learning (RL)
│             ├── Basics: agent, environment, reward, policy, value function
│             ├── Algorithms:
│             │       ├── Q-Learning, SARSA
│             │       ├── Deep Q-Networks (DQN)
│             │       ├── Policy Gradients (REINFORCE)
│             │       ├── Actor–Critic (A2C, A3C)
│             │       └── Advanced: PPO, DDPG, SAC
│             ├── Tools: OpenAI Gym, Stable Baselines3
│             └── Projects: game-playing agents, simple robotics tasks
│
├── Deep Learning (DL)
│     │
│     ├── Neural Networks (ANN)
│     │       ├── Perceptron → MLP → architecture design
│     │       ├── Activation functions (ReLU, LeakyReLU, ELU, Sigmoid, Tanh)
│     │       ├── Loss functions (cross-entropy, MSE)
│     │       ├── Backpropagation & optimizers (SGD, Adam, RMSprop)
│     │       └── Training practices (early stopping, checkpointing)
│     │
│     ├── Convolutional Neural Networks (CNN)
│     │       ├── Convolution, stride, padding, pooling
│     │       ├── Architectures: LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, MobileNet
│     │       ├── Tasks: image classification, object detection, segmentation
│     │       ├── Transfer learning & fine-tuning
│     │       └── Data augmentation (rotate, flip, crop, color jitter)
│     │
│     ├── Recurrent Neural Networks (RNN)
│     │       ├── Sequence modelling basics
│     │       ├── LSTM, GRU internals & gate logic
│     │       ├── Sequence-to-sequence (seq2seq) models
│     │       └── Attention mechanism (leading to Transformers)
│     │
│     ├── Transformers
│     │       ├── Self-attention, multi-head attention
│     │       ├── Encoder / Decoder stacks
│     │       ├── Popular models: BERT (encoder), GPT (decoder), T5 (encoder-decoder)
│     │       └── Vision Transformers (ViT) for image tasks
│     │
│     └── Generative Models
│             ├── GANs: generator, discriminator, training stability tricks
│             ├── VAEs: latent space, reconstruction loss
│             └── Diffusion models: basics and image synthesis
│
├── Natural Language Processing (NLP)
│     │
│     ├── Basics & Preprocessing:
│     │       ├── Tokenization (word, subword/BPE, SentencePiece)
│     │       ├── Stopwords, stemming, lemmatization
│     │       ├── Text normalization, handling emojis/HTML
│     │       └── Vectorization: TF-IDF, CountVectorizer
│     │
│     ├── Embeddings:
│     │       ├── Word2Vec, GloVe, FastText
│     │       └── Contextual embeddings: ELMo, BERT embeddings
│     │
│     ├── Core tasks:
│     │       ├── Text classification (sentiment, intent)
│     │       ├── Sequence labeling (NER, POS tagging)
│     │       ├── Machine translation
│     │       ├── Text summarization (extractive & abstractive)
│     │       ├── Question Answering
│     │       └── Language generation (LLMs)
│     │
│     ├── Language Models (LMs):
│     │       ├── n-gram models (baseline)
│     │       ├── RNN/LSTM based LMs
│     │       └── Transformer LMs (pretraining & fine-tuning)
│     │
│     └── Tools & Libraries:
│             ├── Hugging Face Transformers
│             ├── spaCy, NLTK
│             └── sentence-transformers
│
└── Computer Vision (CV)
      │
      ├── Image Processing (low-level ops)
      │       ├── Filtering (Gaussian, Median)
      │       ├── Edge detection (Sobel, Canny)
      │       ├── Morphological ops (erosion, dilation)
      │       └── Thresholding & color-space conversions
      │
      ├── Image Classification
      │       ├── CNNs, transfer learning, data augmentation
      │       └── Datasets: CIFAR-10/100, ImageNet
      │
      ├── Object Detection
      │       ├── Two-stage: Faster R-CNN
      │       ├── Single-stage: YOLO, SSD
      │       ├── Metrics: mAP, IoU
      │       └── Datasets: COCO, Pascal VOC
      │
      ├── Image Segmentation
      │       ├── Semantic: UNet, DeepLab
      │       ├── Instance: Mask R-CNN
      │       └── Metrics: IoU, Dice coefficient
      │
      ├── Face Recognition
      │       ├── Face detection (MTCNN, Haar, DNN)
      │       ├── Face embedding models (FaceNet, ArcFace)
      │       └── Tasks: verification, identification
      │
      └── Vision Transformers (ViT)
              ├── Patch embedding, positional encoding
              └── When to use vs CNNs

