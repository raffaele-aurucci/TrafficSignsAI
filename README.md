# An Efficient Federated Learning Method for Damaged Road Sign Detection in Smart Cities

## Scope

**Road safety** is a growing priority for public administrations worldwide. Traditional
**manual inspection of road signage** is costly and inefficient — the **smart city paradigm**
combined with **AI** offers a viable path to automate this process.

This project proposes a **distributed pipeline for road sign detection and classification**
optimized for **edge devices**, integrating three modules:

- **Object Detection (client-side)** — lightweight YOLO models for traffic sign detection
- **Federated Learning (server-side)** — distributed training of classifiers for sign condition (damaged / healthy)
- **Data Pruning (client-side)** — influence-score-based sample selection on the last model layer

Experiments on the **Mapillary Traffic Sign Dataset (MTSD)** show that filtering small
bounding boxes significantly boosts detection — **YOLO26s reaches F1 = 0.7214** vs 0.6326
without filtering. In the federated setting, **Vision Transformers** emerge as the best
trade-off between efficiency, robustness, and accuracy. Data pruning reduces training
time by **up to 22%** while keeping performance loss **within 2 percentage points**.

### Pipeline for Detection & Classification Road Signs

<img src="plots/pipeline.png" width="100%"/>

The system follows a **distributed pipeline** where **edge devices** and the **central server** collaboratively train the model using **Federated Learning**.

1. **Image Acquisition:** *Edge devices* capture images from the environment.

2. **Object Detection:** A *lightweight Object Detector* identifies *traffic signs* in the images.

3. **ROI Extraction & Preprocessing:** The detected *Regions of Interest (ROIs)* are extracted and processed through a *preprocessing stage*.

4. **Global Model Distribution:** The server sends the *updated global model* to the *edge devices*.

5. **Data Pruning:** Each device performs *data pruning* to select the *most informative samples* for training.

6. **Local Training:** Using *Federated Learning*, each device trains the classifier locally on its *pruned dataset*.

7. **Model Upload:** After local training rounds, model parameters are sent to the server.

8. **Aggregation:** The server *aggregates the parameters* to produce an *updated global model*.

9. **Model Redistribution:** The new global model is redistributed to the edge devices to repeat the training cycle.


## Data Pruning Module

The **data pruning module** is executed **locally on each client before training begins**
to remove **noisy, mislabeled, or redundant samples**, improving the convergence of the
**federated learning process**.

For each training sample $(x_i, y_i)$, an **influence score** is computed as the
**L2 norm of the gradient of the loss with respect to the last linear layer** of the model:

$$s_i = \left\| \nabla_{\theta_L} \mathcal{L}(f_\theta(x_i), y_i) \right\|_2$$

where $\theta_L$ includes both the **weights $W$** and **bias $b$** of the last layer,
concatenated into a single vector. This score quantifies how much the sample
**influences the model update**.

The scores are then **normalized per class using z-score normalization**:

$$z_i = \frac{s_i - \mu_c}{\sigma_c}$$

where $\mu_c$ and $\sigma_c$ are the mean and standard deviation of the influence scores
for class $c$. Only samples with **normalized scores within the threshold interval**
$z_i \in [-\varepsilon, +\varepsilon]$ are retained. This removes both:

- **High-gradient outliers** — potentially noisy or mislabeled samples
- **Low-gradient samples** — redundant or uninformative samples

A **class safeguard mechanism** ensures that **each class retains at least a minimum
number of samples**, reintegrating the most representative ones if necessary.

### Experimental Results on Training Time
The results reported in the table represent the **median time gain** aggregated across
all grid search configurations.
The data confirm a clear trend: **lower learning rates yield higher time savings**,
with **LR = 0.0001** consistently producing the largest gains across all models —
reaching up to **+17.5%** (MobileViT Small) and **+14.5%** (ViT Tiny).
At higher learning rates the benefit shrinks, and in some cases turns negative,
likely due to training instability in the post-pruning phase rather than a true
increase in computational cost.

![](plots/training_time_lr_x_threshold.png)

### Experimental Results on F1-Score
In most cases, pruning introduces **negligible F1-score degradation (< 1%)** across
all aggregation algorithms and thresholds. The main exception is **ResNet18**, which
suffers the largest drops — up to **−4.92%** with FedLC at threshold 0.5 — consistent
with its higher dataset reduction rate. **FedProx** tends to amplify degradations
slightly, likely due to the interaction between its proximal regularization term and
the gradient-based pruning criterion.

![](plots/f1_by_alg_threshold.png)
