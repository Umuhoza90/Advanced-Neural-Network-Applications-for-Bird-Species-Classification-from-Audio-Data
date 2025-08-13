# Advanced-Neural-Network-Applications-for-Bird-Species-Classification-from-Audio-Data

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Methodology](#methodology)
- [Computational Results](#computational-results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)
---

## Abstract

This study explores the use of convolutional neural networks (CNNs) for classifying bird species based on audio spectrograms, utilizing data sourced from the Xeno-Canto database. The research focuses on developing two models: a binary classifier to distinguish between American Crow ('amecro') and Steller's Jay ('stejay'), and a multi-class classifier to identify twelve different bird species: American Crow ('amecro'), Barn Swallow ('barswa'), Black-capped Chickadee ('bkcchi'), Blue Jay ('blujay'), Dark-eyed Junco ('daejun'), House Finch ('houfin'), Mallard ('mallar3'), Northern Flicker ('norfli'), Red-winged Blackbird ('rewbla'), Steller's Jay ('stejay'), Western Meadowlark ('wesmea'), and White-crowned Sparrow ('whcspa').
Our approached involved converting raw audio recordings into spectrograms, normalizing, and padding them to ensure uniform input dimensions. The binary classification model achieved near-perfect accuracy, demonstrating its effectiveness in distinguishing between the two target species. For the multi-class classification task, Model 2, featuring a deeper network with additional convolutional and dense layers, achieved perfect accuracy on both training and validation datasets, indicating its robust learning and generalization capabilities.
The results showcase the binary model achieving near-perfect accuracy, while the multi-class model demonstrates substantial learning capabilities, highlighting the effectiveness of CNNs in bioacoustic applications.

[Back to top](#table-of-contents)

---

## Introduction

Bird vocalization analysis is a critical component of ecological monitoring, traditionally reliant on expert knowledge and often prone to human error. Accurate identification of bird species through their calls is essential for understanding biodiversity, tracking migration patterns, and monitoring environmental health. However, manual analysis is time-consuming and may lack consistency, highlighting the need for automated solutions.

This study leverages the power of Convolutional Neural Networks (CNNs), which have demonstrated remarkable success in image recognition tasks, to automate and enhance the accuracy of bird species classification from audio recordings. By transforming audio data into spectrogram images, CNNs can efficiently classify bird calls. This approach enables the extraction and learning of complex patterns from visual representations of audio signals, crucial for accurate species identification.

The importance of this research lies in its potential to significantly improve the reliability and scalability of bird species identification, providing a robust tool for ecological research and conservation efforts. The scientific aim of this study is to develop and evaluate two CNN-based models: a binary classifier to distinguish between American Crow (Corvus brachyrhynchos) and Steller's Jay (Cyanocitta stelleri), and a multi-class classifier to identify twelve different bird species: American Crow (Corvus brachyrhynchos), Barn Swallow (Hirundo rustica), Black-capped Chickadee (Poecile atricapillus), Blue Jay (Cyanocitta cristata), Dark-eyed Junco (Junco hyemalis), House Finch (Haemorhous mexicanus), Mallard (Anas platyrhynchos), Northern Flicker (Colaptes auratus), Red-winged Blackbird (Agelaius phoeniceus), Steller's Jay (Cyanocitta stelleri), Western Meadowlark (Sturnella neglecta), and White-crowned Sparrow (Zonotrichia leucophrys). By addressing these aims, this research contributes to the growing body of literature on bioacoustic monitoring and demonstrates the applicability of advanced machine learning techniques in ecological studies.

[Back to top](#table-of-contents)

---

## Theoretical Background

CNNs are particularly suited for tasks that involve large amounts of image data due to their ability to perform feature extraction directly from images, making them ideal for interpreting spectrograms—visual representations of sound. This project utilizes two neural network architectures tailored to different classification challenges: a binary model and a multi-class model. The binary model employs a sigmoid activation function to distinguish between two species, optimizing binary cross-entropy loss. In contrast, the multi-class model uses a softmax function to handle classifications across twelve species, aiming to minimize categorical cross-entropy loss. Example figures of Neuro Network diagrams.

Figure 1: Neural Network diagram with a single hidden layer
<img width="725" height="482" alt="image" src="https://github.com/user-attachments/assets/ed8005a9-bb5a-48e3-b416-18a3bec0f9a6" />

This figure shows a simple feed-forward neural network for modeling a quantitative response using p = 4 predictors. In the terminology of neural networks, the four features X1, X2, X3, and X4 make up the units in the input layer. The arrows indicate that each of the inputs from the input layer feeds into each of the K hidden units.

<img width="693" height="144" alt="image" src="https://github.com/user-attachments/assets/9203ec39-87c5-47c4-b799-72a2e79631a2" />

Modern neural networks typically have more than one hidden layer, and often many units per layer.A single hidden layer with a large number of units has the ability to approximate most functions.The preferred choice in modern neutral networks is the ReLU (rectified linear unit) activation function. Its activation can be computed and stored more efficiently than a sigmoid activation.Modern neural networks typically have more than one hidden layer, and often many units per layer(figure 2) [1]. In theory a single hidden layer with a large number of units has the ability to approximate most functions. However, the learning task of discovering a good solution is made much easier with multiple layers each of modest size.

Figure 2: Neural Network diagram with two hidden layers and multiple outputs

<img width="1017" height="847" alt="image" src="https://github.com/user-attachments/assets/dcf5d153-07cb-4982-aaf0-dc6d5581cc22" />


[Back to top](#table-of-contents)

---

## Methodology

The dataset comprised spectrograms generated from audio clips, which were then normalized and padded to ensure uniform input dimensions. The padding process standardized the spectrograms by ensuring each matched the largest time dimension across all samples, facilitating consistent input sizes for CNN processing. Raw audio recordings were converted into spectrograms to capture the distinct frequency and time characteristics of each bird call. These spectrograms were then resized by padding to match the largest time dimension observed within the dataset. For spectrograms with a time dimension less than the maximum, zeros were added along the time axis until the maximum length was reached. This method preserved the integrity of the original audio data while standardizing the input size for the neural network. Additionally, normalization was applied to each spectrogram, scaling the pixel values to a range between 0 and 1 to facilitate faster convergence during network training. The dataset was subsequently partitioned into training and validation sets, comprising 80% and 20% of the data, respectively, allowing for effective training and performance validation of the models.

The binary classification model specifically aimed to distinguish between American Crow ('amecro') and Steller's Jay ('stejay'). The spectrograms were labeled '0' for American Crow and '1' for Steller's Jay, creating a clear binary target for model training. For the multi-class classification task, the focus was on classifying all twelve bird species identified in the dataset: American Crow ('amecro'), Barn Swallow ('barswa'), Black-capped Chickadee ('bkcchi'), Blue Jay ('blujay'), Dark-eyed Junco ('daejun'), House Finch ('houfin'), Mallard ('mallar3'), Northern Flicker ('norfli'), Red-winged Blackbird ('rewbla'), Steller's Jay ('stejay'), Western Meadowlark ('wesmea'), and White-crowned Sparrow ('whcspa').

To preprocess the test MP3 data, the following steps were taken: Firstly, the test MP3 files were loaded using Librosa, and the Short-Time Fourier Transform (STFT) parameters were set to define the frequency and time resolution. The energy of each frame was calculated to identify high-energy segments of the audio. High-energy segments were identified based on an energy threshold. For each identified segment, a 2-second audio clip was extracted, and a spectrogram was generated using the STFT. The spectrograms were then resized to have consistent dimensions (256 frequency bins and 343-time steps) using bilinear interpolation to match the input size expected by the CNN. The resized spectrograms were saved into an HDF5 file for subsequent model prediction, with each spectrogram stored with a unique identifier to facilitate easy retrieval. The spectrograms were then normalized by scaling the pixel values to a range between 0 and 1, and a channel dimension was added to match the input shape required by the CNN.

After evaluating multiple neural network models with different architectures and hyperparameters for the multi-class classification task, Model 2 emerged as the best choice. Model 2, a deeper network with additional convolutional and dense layers, achieved perfect accuracy on both the training and validation datasets, demonstrating its ability to effectively learn and generalize the patterns in the data. It also had the lowest final training and validation loss values, indicating high confidence in its predictions. In contrast, the baseline model (Model1) and the model with dropout regularization (Model 3) also performed well but had slightly higher loss values. The model using the SGD optimizer (Model 4) showed significantly lower performance, with reduced accuracy and higher loss. Based on these results, Model 2's superior accuracy and minimal loss make it the most robust and reliable model for accurately classifying the 12 bird species in our dataset. Both the binary and multi-class models utilized convolutional layers for feature extraction, followed by max-pooling layers to reduce dimensionality and enhance feature extraction. Dense layers followed the convolutional and pooling layers, with a sigmoid activation function used for the binary model and a softmax activation function for the multi-class model, each tailored to the specific classification needs of the tasks.

[Back to top](#table-of-contents)

---

## Computational Results

<img width="921" height="386" alt="image" src="https://github.com/user-attachments/assets/98cccdc9-441e-41a7-bb65-c5aded210d8d" />

<img width="935" height="373" alt="image" src="https://github.com/user-attachments/assets/41deb979-4282-42c5-ade8-1850829c889a" />

<img width="990" height="710" alt="image" src="https://github.com/user-attachments/assets/6ca13b41-829d-498b-9fa6-f1873306dc85" />

<img width="967" height="763" alt="image" src="https://github.com/user-attachments/assets/0c41af03-d97f-4c51-beb4-27d25fef5e0b" />

<img width="1137" height="598" alt="image" src="https://github.com/user-attachments/assets/604d09a8-11cc-45f2-839e-9140264de6ea" />

<img width="1144" height="615" alt="image" src="https://github.com/user-attachments/assets/0b9c7aa3-f4d7-4f6d-885a-3dd91efa6f35" />

<img width="1172" height="543" alt="image" src="https://github.com/user-attachments/assets/37519141-581c-4e95-9e85-233e82d280cd" />

<img width="1157" height="507" alt="image" src="https://github.com/user-attachments/assets/b3492703-007c-486c-927c-e4857039ad8e" />

<img width="1145" height="462" alt="image" src="https://github.com/user-attachments/assets/ec1c5d30-d760-4b37-861d-50c920e4700f" />

[Back to top](#table-of-contents)

---

## Discussion

The binary model’s rapid achievement of near-perfect accuracy demonstrates its significant potential utility in targeted ecological monitoring tasks, where specific species differentiation is crucial. The model successfully distinguished between American Crow and Steller's Jay, achieving high accuracy and minimal loss on both training and validation datasets. This result suggests that the binary model captured the distinguishing features of the two bird species effectively, highlighting the robustness and reliability of CNNs for binary classification tasks in bioacoustic applications.

However, the multi-class model presented more complex challenges. Despite this, Model 2 emerged as the best-performing architecture among the evaluated models, achieving perfect accuracy on both training and validation datasets. The deep architecture of Model 2, featuring additional convolutional and dense layers, enabled it to effectively learn and generalize the patterns in the data. The model demonstrated the lowest final training and validation loss values, indicating high confidence in its predictions. This performance underscores the effectiveness of a well-designed CNN architecture in handling multi-class classification tasks, even with diverse and complex datasets.

The baseline model (Model 1), the model with dropout regularization (Model 3), and the model using the SGD optimizer (Model 4) were all multi-class models designed to predict the following 12 bird species: American Crow ('amecro'), Barn Swallow ('barswa'), Black-capped Chickadee ('bkcchi'), Blue Jay ('blujay'), Dark-eyed Junco ('daejun'), House Finch ('houfin'), Mallard ('mallar3'), Northern Flicker ('norfli'), Red-winged Blackbird ('rewbla'), Steller's Jay ('stejay'), Western Meadowlark ('wesmea'), and White-crowned Sparrow ('whcspa'). Among these models, Models 1 and 3 performed well but had slightly higher loss values compared to Model 2, indicating slightly less confidence in their predictions. Model 4 showed significantly lower performance, with reduced accuracy and higher loss. This disparity highlights the importance of selecting appropriate optimizers and model architectures to achieve optimal performance in neural network training.

While both models achieved high validation accuracies, suggesting effective capture of distinguishing features of each bird species, the journey to these results highlighted several critical aspects. The binary model’s rapid convergence to high accuracy indicates that distinguishing between two species is relatively straightforward for CNNs, provided the data is well-preprocessed and labeled. In contrast, the multi-class model's development underscored the inherent challenges in broader bioacoustic applications. The need for comprehensive training and carefully tuned network architectures became evident to handle the diversity within the dataset.

The pre-trained Keras model, best_bird_species_classifier.keras, was successfully loaded and used to predict bird species from external test audio files (test1.mp3, test2.mp3, and
test3.mp3). The audio files were processed by converting them to NumPy arrays, detecting loud segments based on an energy threshold, and generating Mel spectrograms for these segments. The spectrograms were resized and normalized before being fed into the model for predictions. The model consistently predicted "wesmea" (Western Meadowlark) for all spectrograms with a probability of 1.0, suggesting a strong bias or overfitting towards this class. This could be due to an imbalance in the training data or feature similarities in the audio clips. Further investigation, including re-training with a balanced dataset and thorough model evaluation, is recommended to address this bias.
Challenging Species and Confusions: The model consistently predicted "wesmea," indicating it might be biased or overfitted. Listening to the bird calls and examining the spectrograms, it appears that the model did not differentiate between different bird species. This suggests that other species' calls were either too similar to the "wesmea" call in the feature space the model learned or that the model was not exposed adequately to diverse examples during training.

Alternative Models and Neural Network Suitability: Other models that could be used for this task include Support Vector Machines (SVM), Random Forests, and Gradient Boosting Machines (GBM). However, neural networks, especially Convolutional Neural Networks (CNNs), are well-suited for this application because they can effectively capture and learn from the complex patterns in spectrograms, which are essentially images representing the frequency content of the audio signals over time. Neural networks can automatically learn hierarchical features from the data, making them powerful for tasks involving image-like data, such as spectrograms in audio classification.

[Back to top](#table-of-contents)

---

## Conclusion

This study evaluates the use of Convolutional Neural Networks (CNNs) for classifying bird species using audio spectrograms from the Xeno-Canto database. Two models were developed: a binary classifier to distinguish between American Crow and Steller's Jay, and a multi-class classifier to identify twelve bird species. The binary model achieved near-perfect accuracy, while the multi-class model, particularly the deeper Model 2, achieved perfect accuracy on both training and validation datasets.

The process involved converting audio recordings into spectrograms, normalizing, and padding them to ensure uniform input dimensions. The binary model's high accuracy demonstrates CNNs' effectiveness in specific species differentiation. In contrast, the multi-class model successfully handled the complexity of identifying multiple species, proving the robustness of well-designed CNN architectures.

However, challenges were noted, such as the model's consistent prediction of "wesmea" (Western Meadowlark) for all test samples, indicating potential biases or overfitting. This issue may stem from imbalanced training data or insufficient exposure to diverse examples, necessitating further investigation and re-training with a balanced dataset.
In conclusion, this research highlights the potential of CNNs in bioacoustic applications, offering reliable and scalable methods for bird species identification. The findings contribute to
ecological monitoring and conservation efforts, providing valuable tools for understanding and preserving biodiversity. Future work should focus on refining these models, addressing biases, and exploring additional architectures to improve accuracy and reliability in automated bird call classification systems.

[Back to top](#table-of-contents)

---

## References
1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical
learning. Retrieved from https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html

[Back to top](#table-of-contents)

---
