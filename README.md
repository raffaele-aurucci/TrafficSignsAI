# An Efficient Federated Learning Method for Damaged Road Sign Detection in Smart Cities

## Introduction

**Road safety** is an increasingly important priority for public administrations worldwide. In this context, the proper maintenance of **vertical road signage** plays a crucial role. However, it becomes **costly and inefficient** when relying on traditional **manual inspection methods**. The **smart city paradigm**, combined with recent advances in **Artificial Intelligence (AI)**, offers new opportunities to **automate this process**.

This **project** proposes and evaluates a **distributed pipeline for road sign detection and classification**, optimized for **deployment on edge devices**. The architecture integrates **three main modules**:

- **Object Detection module (client-side)** – uses **lightweight YOLO models** to detect traffic signs.
- **Federated Learning module (server-side)** – coordinates the **distributed training of classifiers** that determine the **condition of the sign (damaged or healthy)**.
- **Data Pruning module (client-side)** – selects the **most informative samples** using **influence scores** computed on the **last layer of the global model**.

Experiments conducted on the **Mapillary Traffic Sign Dataset (MTSD)** show that **preliminary filtering of small bounding boxes** significantly improves the performance of lightweight models. 
In particular, **YOLO26s** achieves: **mAP@50 = 0.7450** and **F1-score = 0.7214** compared to **0.6085 mAP@50** and **0.6326 F1-score** obtained **without filtering**.

In the **Federated Learning setting**, three architectural families were evaluated:

- **Convolutional Neural Networks (CNN)**
- **Vision Transformers (ViT)**
- **Hybrid models (CNN + ViT)**

using different **aggregation strategies** and **pruning thresholds**. Among the tested models, **Vision Transformers (ViT)** emerge as the **best trade-off between computational efficiency, pruning robustness, and overall accuracy**. 

The results demonstrate that targeted **data pruning** can **reduce training time by up to 22%**, while maintaining **predictive performance within 2 percentage points**.

## Pipeline for Detection & Classification Road Signs

