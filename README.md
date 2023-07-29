# AML Project 2022 

Practical programming projects in the course Advanced Machine Learning 

## Task 0: Demo task to get familiar with the platform

Sample mean on the training set is used as the prediction for the test set.

## Task 1: Predict a person's age from brain MRI images/features

We divided the process in two steps:
- preprocessing of all the data, including outlier detection, feature selection and imputation of missing values
- regression in order to compute the final predictions

Imputation of missing values was done using KNN imputer of sklearn with 3 neighbors.
IsolationForest model dropped outliers from data.
All data were scaled using sklearn.StandarScaler.
The best 166 features were selected using sklearn.SelectPercentile(f_regressor) with percentile value of 10. 
From these 166 we picked the best 40 features through sklearn.SelectKBest(f_regressor).

The remaining 126 features from sklearn.SelectPercentile are squeezed into 60 using a Keras autoencoder with this structure: 
- input layer
- one dropout layer (p=0.2)
- the bottleneck layer with 60 neurons
- output layer

The preprocessed data have 100 features and it's ready for regression models:
- sklearn GradientBoostingRegressor with number of estimators=80
- sklearn GaussianProcessRegressor with Rational Quadratic Kernel, 
- sklearn ExtraTreesRegressor with number of estimators=5000 
- sklearn support vector machine with RBF kernel
Hyperparameters ExtraTreesRegressor and SVM were chosen through GridSearch.

The final output is a StackingRegressor between the 4 regression models to obtain a better generalization.

To overcome randomness of the autoencoder the handed-in submission is the mean of multiple runs.

## Task 2: ECG time-series classification of arrhythmia 

Report: Preprocessing

Data Cleaning and Missing Value Handling: Prior to further processing, each row of the dataset undergoes a thorough cleaning process. Missing values are stripped from the data, ensuring a complete and consistent signal for analysis.

ECG Signal Cleaning: To improve the quality of the data, the ecg_clean function from the neurokit library is utilized. This function effectively filters and denoises the electrocardiogram (ECG) signal, removing unwanted artifacts and noise.

ECG Peaks Extraction: The ecg_peaks function from the neurokit library is applied to the preprocessed signal. This step accurately identifies and extracts the peaks from the ECG signal, capturing crucial cardiac events for subsequent analysis.

Feature Extraction: From the resulting preprocessed signal, essential features are extracted. These features serve as crucial indicators for the classification task and provide valuable insights into the underlying patterns and characteristics of the data.

Feature Scaling: To ensure fair and optimal comparison among features, the extracted features are appropriately scaled using StandardScaler from scikit-learn. Scaling standardizes the features and prevents any undue influence due to differing measurement units.

Classification:

The classification task is approached through a robust ensemble of five distinct classification algorithms, skillfully combined using StackingClassifier from scikit-learn.

CatBoost: The first ensemble algorithm utilizes the powerful CatBoostClassifier from the catboost library. It employs the 'MultiClass' objective, and with 300 iterations, it effectively captures complex relationships and dependencies within the data. The random_state is set to 0 to ensure reproducibility, and the evaluation metric 'TotalF1:average=Micro' ensures a comprehensive assessment of the classifier's performance.

ExtraTrees: The second algorithm in the ensemble is ExtraTreesClassifier from scikit-learn, which employs 1000 estimators. This ensemble of extremely randomized decision trees excels in handling high-dimensional data and contributes to the ensemble's overall predictive accuracy.

Support Vector Machine (SVM): The third algorithm is SVC from scikit-learn, employing the default parameters. SVM is renowned for its ability to handle both linear and non-linear classification tasks effectively.

XGBoost: The fourth algorithm is XGBClassifier from the xgboost library, utilizing 100 estimators. With the 'multi:softmax' objective and the metric 'f1_score,' XGBoost effectively addresses multi-class classification challenges while optimizing performance.

Random Forest: The final algorithm in the ensemble is RandomForestClassifier from scikit-learn, utilizing 300 estimators. This ensemble of decision trees provides robust predictions and contributes to the ensemble's diversity.

By combining the predictions of these diverse classifiers using the StackingClassifier, we leverage the strengths of each model, achieving a more robust and accurate classification outcome.

Overall, this comprehensive approach to data preprocessing and classification ensures that we are equipped to tackle the challenges of the classification task and obtain meaningful insights from the data.

## Task 3: Mitral valve segmentation in video sequences

Implementation of the Task:

We decided to leverage all the videos provided, including both the expert and amateur ones, for our segmentation task. Each video was treated as a sequence of frames. To create a consistent dataset for training, we aggregated all the frames and rescaled them to the same dimensions. We experimented with two resolutions: 112x112 and 224x224.

Data Preprocessing:

a. Frame Rescaling: We rescaled all frames to either 112x112 or 224x224 pixels to ensure uniformity during training. The lower resolution of 112x112 allowed us to obtain a larger number of images, while the higher resolution of 224x224 retained the quality of expert images, enabling better augmentation.

b. Data Augmentation: To increase the diversity and robustness of our dataset, we performed data augmentation using keras.preprocessing.image.ImageDataGenerator. The following augmentation parameters were applied: rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, and zoom_range=0.1. This resulted in approximately 1000 augmented images of size 224x224, providing ample variations for training.

Model Architecture:

We opted to use a U-Net architecture for the segmentation task. The U-Net consists of an encoder (contraction) path and a decoder (expansion) path, connected by a middle layer.

a. Encoder (Contraction): The encoder comprises four layers, which gradually reduce the spatial dimensions of the input while capturing important features.

b. Middle Layer: The middle layer acts as a bottleneck, where the most critical features are captured and represented compactly.

c. Decoder (Expansion): The decoder consists of four layers, which gradually upsample the representation to the original spatial dimensions, reconstructing the segmented output.

Model Training:

We trained the U-Net architecture with early stopping and model checkpointing on a validation set. Early stopping helps prevent overfitting and ensures that the model is saved at the iteration with the best performance on the validation set.

Segmentation of Testing Images:

After selecting the best model based on its performance on the validation set, we used this model to perform segmentation on the testing images. Before segmentation, we resized the testing images to 224x224 to match the resolution used during training.