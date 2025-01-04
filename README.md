[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
<h1> SafeNet: Prototype Lightweight ConvNet for Grid-Based Binary Classification of Safe UAV Landing Zones </h1>

**Rowan Morse - University of Pittsburgh '28** 

|||| 
|-----------------------------|-----------------------------| -----------------------------| 
| ![Gif1](/imgs/gifs/gif1.gif) <br> ![Gif4](/imgs/gifs/gif5.gif) <br> | ![Gif2](/imgs/gifs/gif2.gif) <br> ![Gif5](/imgs/gifs/gif6.gif) <br>| ![Gif3](/imgs/gifs/gif4.gif) <br> ![Gif6](imgs/gifs/gif7.gif) <br>| 
<p align="center" style="font-style: italic; margin-top: 10px;">
  GIFs made from predictions on Test data at a 0.40 confidence threshold. The red dots indicate "unsafe" regions predicted by SafeNet.
</p>


## Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
5. [Dataset & Data-Pipeline](#dataset)
6. [Model Details](#model-details)
    - [Model Architecture](#model-architecture)
    - [Important Metrics & Performance Benchmarks](#important-metrics-&-benchmark-performance-goals)
    - [Prioritizing Recall](#prioritizing-recall-weighted-binary-crossentropy-loss)
        1. [Standard Binary Cross-Entropy Loss](#standard-binary-crossentropy-loss)
        2. [Class Weights to Improve Recall Specifically](#class-weights-to-improve-recall-specifically)
        3. [Weighted Binary Cross-Entropy Loss](#weighted-binary-crossentropy-loss)
7. [Results](#performance-metrics)
    - [Test Data Examples](#test-data-examples)
    - [Test Benchmarks](#test-benchmarks)
    - [Confusion Matrix of Test Predictions](#confusion-matrix-of-test-predicitons)
8. [Future Improvements](#future-improvements)
    - [Fighting Fires/Other Uses](#fighting-firesother-uses)
    - [Further Improve Latency](#further-improve-latency)
    - [Ensemble](#ensemble)

---

## Overview  
SafeNet is a lightweight convolutional neural network designed for compatibility with edge devices to predict safe landing zones for drones. Using a grid-based binary classification approach, it analyzes image sections to determine areas free from hazards like water, buildings, and trees.

**Why is this significant?** 
With an increased use of UAVs (unmanned aerial vehicles) in **disaster relief and warfare,** operators may encounter disconnections caused by weather, low battery, obstacles, or **GPS Jammers** and in these scenarios a protocol to land safely by just taking an image of the ground below. 

This project was my way of introducing myself to computer vision and by no means do I think this model is deployable, but I do believe that similar approach/models could solve real world problems like **aid/package delivery** or even **analyze/fight wild-fires** using the models predictions of flat areas which could help identify safe zones for landing, optimize water bomber flight paths, or assist in assessing areas for safe operations during emergencies.

---

## Key Features  
- **Edge-Capable:** Model inferences based on entire image allowing for real time predictions on edge devices suitable for quadcopter drones. While I was unable to test my model on any industry edge devices, sub 1-second predictions ran on my Intel i7 CPU. [This article](https://medium.com/@samsterckval/google-coral-edge-tpu-vs-nvidia-jetson-nano-a-quick-deep-dive-into-edgeai-performance-bc7860b8d87a) found that the NVIDIA Jetson Nano performs similar if not better than the i7–4870HQ on machine learning tasks because of its specialized GPU architecture, which is optimized for parallel processing in machine learning tasks.

- **Grid-Based Labeling:** Divides large images into 4800 (60x80) binary labels corresponding to pixel safety (0 = "safe", 1 = "unsafe") 
- **Lightweight Model Architecture:** YOLO inspired, the model predicts the entire image rather than image kernels allowing for faster predictions compared to the alternative sliding window approach.

- **Class Weights to Prioritize Safety:** By weighting safe predictions more than unsafe predictions the model is trained to prioritize recall and limit false-safe predictions. This feature is essential for deployment as losing a drone could be costly, so the model must have very high confidence for safe predictions. 
## Dataset & Pipeline 
### Dataset: FloodNet

The **FloodNet dataset** is a high-resolution UAV imagery dataset designed for post-disaster damage assessments. The images were captured after Hurricane Harvey which caused major flooding in suburban areas producing an abundance of obstacles that can be utilized during training simulating disaster relief scenarios.

#### Key Features of FloodNet:
- **10 Classes** ('Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9)
- **Total Images:** 2343 (Train: 1445, Val: 450, Test: 448)
- **High-Resolution UAV Imagery:** Provides detailed 3000x4000px images taken at a height of 200 feet with a high 1.5cm spatial resolution.  
- **Semantic Segmentation:** Pixel-wise labeling allows masks that can be used for regional classification. 


#### How My Project Uses FloodNet:
Firstly, using the original segmentation labels from the masks I needed to separate the labels into "Safe" or "Unsafe" for classification. 

Of the 10 labels I concluded that...

Flat areas like **grass and roads/parking lots** could be considered **safe** for drone landing. I originally labeled the road unsafe but later revised as most disaster scenarios that would require drone deployment would be due to road inaccessibility, meaning that there would be no moving cars to avoid. 

I deemed any **background, buildings, flooded-roads, water, trees, pools, or vehicles** as unsafe which is self-explanatory. 

Then using the masks I was able to create 60 row x 80 column binary labels for each image. This was executed by sliding a 50x50px across the masks with a step of 10px if the window contained more than 20% "unsafe" pixels a 1 would be placed in the label corresponding to that region of the original photo. Using region based labels I avoided having to execute a sliding window algorithm for post training inferencing. If I had used a sliding window post-training I would've had to make the model predict every region of an image which causes far more latency in terms of runtime and is far less effective than classifying the entire image.


#### Accessing the Dataset:
The FloodNet dataset is publicly available and can be downloaded from the [official repository](https://github.com/BinaLab/FloodNet-Supervised_v1.0).

### Data Pipeline 
<p align="center">
  <img src="imgs/diagrams/pipeline_chart.png" alt="Flow Chart of Data Pipeline" width="600"/>
</p>


1. First, I took the original 3000x4000px FloodNet dataset and for each image/mask I resized down to 600x800px. 
2. I then used the Albumentations library to **5x the amount of training data** using data augmentation techniques like random flips/rotations, random lighting, etc. 
3. Then taking the resize/augmented images and masks I ran the previously discussed sliding window to generate labels for each image that were stored in CSV files making them easily accessible. 

## Model Details 
### Model Architecture 
<p align="center">
  <img src="imgs/diagrams/arch.png" alt="SafeNet Architecture" width="600"/>
</p>
<p align="center">
  <i>SafeNet Architecture Diagram</i><br>
  Input = (600x800x3)<br>
  Purple = Conv2D, ReLU <br>
  Red = MaxPool, (2x2), stride = 2 <br>
  Pink = Pointwise, (1x1), ReLU<br>
  Green = GlobalPool<br>
  Yellow = Sigmoid
</p>


#### Key Features 

1. **Light Weight** 
    - The general goal of the model architecture was to minimize runtime for forward pass to the model. While designing, I often tested for FLOPs (floating-point operations per second) to gain general insight into possible model performance, despite the metrics' limitations. 
    - One may notice that the architecture is very similar to **YOLOv1**, aside from minor dimensional/layer changes, as YOLO has been shown to perform well on bounding-box object detection. Considering this task requires a similar approach, with the difference being the model isn't predicting boxes then performing Non-Max Suppression (NMS), but instead the model is classifying the image cells. 
    

2. **Pooling, Downsampling, & Pointwise Convolutions** 
    - The model uses **Max Pooling, strided convolutions, and pointwise(1x1) convolutions** for spatial reduction, minimizing computational overhead while preserving critical information.

3. **Grid-Based Prediction**  
- As previously mentioned, the 4800x1 sigmoid output layer is essentially a 60x80 grid of 10x10 regions of the image where each value in the output vector corresponds with the model's confidence of whether the region is "unsafe." 

4. **Note** 
- The current model is **not** yet pruned or quantized which leaves extra room for future runtime advancements. 

### Important Metrics & Performance Benchmarks.

1. **Recall** 
    - **Most Important, Goal >95%** 
    - Measures the proportion of actual unsafe regions that are correctly identified.
    - Important for minimizing false negatives (e.g., missing an unsafe region).
    - Losing a drone is expensive so we want to maximize our models predictions of unsafe regions, recall gives us this insight. 
2. **Precision**
    - Indicates the proportion of predicted unsafe regions that are actually unsafe.
3. **F1 Score** 
    - **2nd Priority, Goal >80%** 
    - The harmonic mean of precision and recall.
    - Balances the trade off between recall and precision, without, our model would predict every region as "unsafe" to maximize recall.  
    - Since our task prioritizes recall, an F1 score of greater than 80% will indicate that our model has a decent ratio for unsafe/safe predictions while still allowing high recall. 
4. **Binary Accuracy**
    - Measures the percentage of correct predictions across all grid regions.
    - Gives general measure of model performance.

Note that all these metrics are influenced by adjustments to the confidence threshold. 


### Prioritizing Recall (Weighted Binary Cross-Entropy Loss)
#### Standard Binary Cross-Entropy Loss
$$
\text{Binary Cross-Entropy Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right)
$$

Where:
- \( N \) is the number of samples.
- \( y_i \) is the true label (either 0 or 1) for sample \( i \).
- \( p_i \) is the predicted probability that the label is 1 for sample \( i \).

#### Class Weights to Improve Recall Specifically
- Incorporating a larger weight to unsafe predictions makes misclassifying an unsafe region as safe becomes costlier to our loss function during training. 
- This pushes the model to better predict unsafe regions, directly improving recall. 

#### Weighted Binary Cross-Entropy Loss
$$
\text{Weighted Binary Cross-Entropy Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( w_1 \cdot y_i \cdot \log(p_i) + w_0 \cdot (1 - y_i) \cdot \log(1 - p_i) \right)
$$

Where:
- \( N \) is the number of samples.
- \( y_i \) is the true label (either 0 for safe or 1 for unsafe) for sample \( i \).
- \( p_i \) is the predicted probability that the label is 1 (unsafe) for sample \( i \).
- \( w_1 \) and \( w_0 \) are the class weights for unsafe and safe regions, respectively.

**SafeNet was trained with weights w0 = 1 & w1 = 1.5** 

## Results 
### Test Data Examples

|Water & Buildings  |Road & Narrow River| 
|-----------------------------|-----------------------------|
| ![Results 1a](imgs/results/results.1a.png) <br> ![Results 1b](imgs/results/results.1b.png) <br> | ![Results 2a](imgs/results/results.5a.png) <br> ![Results 2b](imgs/results/results.5b.png) <br>| 

|Houses & Water| Car, Water, Road, & Trees| 
|-----------------------------|-----------------------------|
| ![Results 1a](imgs/results/results.4a.png) <br> ![Results 1b](imgs/results/results.4b.png) <br> | <br><br> ![Results 3a](imgs/results/results.9a.png) <br> ![Results 3b](imgs/results/results.9b.png) <br> *Misses car, but correctly spots trees <br>not present in original label! |

| Houses & Woods| Tree & Path  | 
|-----------------------------|-----------------------------|
![Results 2a](imgs/results/results.7a.png) <br> ![Results 2b](imgs/results/results.7b.png) <br> *Displays models safety focus with <br> priority on recall| ![Results 3a](imgs/results/results.10a.png) <br> ![Results 3b](imgs/results/results.10b.png) <br> <br> <br> |


### Test Benchmarks 
- Since the confidence threshold drastically affects the models benchmark performance, we can create a visual plot of the metrics throughout different thresholds to gain better insight. 
<p align="center">
  <img src="imgs/diagrams/Grid-Based Binary Classification Metrics vs. Threshold.png" alt="threshold chart" width="600"/>
</p>

- Based on the plot one can decide which threshold suits there needs, or in other words how much risk one is willing to take. 
- For the purpose of further evaluation, I will proceed with a threshold of 0.2 as it has elevated recall with a good precision tradeoff. 

**On Test Data (no TTA,threshold = 0.2)**
- **Recall** = 0.9651 
- **F1 Score** = 0.7839
- **Binary Accuracy** = 0.7455 

As far as **latency**, I have been unable to run inference using SafeNet on any industry grade edge devices, but running on my Intel I7 cpu the model was able to run predictions in around 1 second without any pruning or quantization. 

### Confusion Matrix of Test Predictions
<p align="center">
  <img src="imgs/diagrams/confusionmatrix.png" alt="Confusion Matrix" width="600"/>
</p>


## Future Improvements 
### Fighting Fires/Other Uses 
- As stated, a similar model that is able to classify flat areas could have potential/more impactful uses than safe drone landing. Specifically, a drone could be deployed over a wild fire area and map out a scene from above by identifying flat road/grass regions that would give optimal insight for extinguishing the fire. 

### Further Improve Latency 
- An ambitious goal would be to run 30 inferences per second, but without the ability to further test the ablities of SafeNet in a deployment scenario this is currently unachievable. However, any future improvements to the model would be primarily focused on latency. 

### Ensemble 
- Another idea would be to implement an ensemble of multiple models that would collaborate to classify regions. SafeNet currently underperforms identifying cars which is most likely because of the square anchor box shape; since cars are longer than they are wide they have a more rectangular shape making it harder for the model to identify using square grid kernels. An ensemble of models could solve this by each model having a different output shape, allowing some to be better suited at identifying particular objects. 

[contributors-shield]: https://img.shields.io/github/contributors/morse-rowan/SafeNet
[contributors-url]: https://github.com/morse-rowan/SafeNet/graphs/contributors
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/rowan-morse/











    












