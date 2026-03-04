# SemCoFormer
Toward Semantic Coherent Visual Relationship Forecasting
---

## Abstract
> Visual Relationship Forecasting (VRF) in video aims to anticipate relations among objects without observing future visual content. 
The task relies on capturing and modeling the semantic coherence in object interactions, as it underpins the evolution of events and scenes in videos. 
However, existing VRF datasets offer limited support for learning such coherence due to noisy annotations in the datasets and weak correlations between different actions and relationship transitions in subject-object pair. Furthermore, existing methods struggle to distinguish similar relationships and overfit to unchanging relationships in consecutive frames.
To address these challenges, we present SemCoBench, a benchmark that emphasizes semantic coherence for visual relationship
forecasting in video. Based on action labels and short-term subject-object pairs, SemCoBench decomposes relationship categories and dynamics by cleaning and reorganizing video datasets to ensure predicting semantic coherence in object interactions. In addition, we also present Semantic Coherent Transformer method (SemCoFormer) to model the semantic coherence with a Relationship Augmented Module (RAM) and a Coherence Reasoning Module (CRM). RAM is designed to distinguish similar relationships, and CRM facilitates the model's focus on the dynamics in relationships.
The experimental results on SemCoBench demonstrate that modeling the semantic coherence is a key step toward reasonable, fine-grained, and diverse visual relationship forecasting, contributing to a more comprehensive understanding of video scenes.
---

## Usage

### 1. Environment Setup

We recommend using Python 3.10+ and PyTorch 1.12+.

```bash
conda create -n semco python=3.10
conda activate semcoformer

pip install -r requirements.txt
```
### 2. Dataset Preparation

The dataset should be organized as follows:
```text
Data/
├── VRP/
│   ├── object_features_resnet/
│   ├── object_features_semantic/
│   ├── object_features_spatial/
│   ├── pair_features_resnet/
│   ├── predicate_features_semantic/
│   ├── VRP_annotation_5_train.txt
│   ├── VRP_annotation_5_test.txt
│   └── ...
├── AGP/
│   ├── object_features_resnet/
│   ├── object_features_semantic/
│   ├── object_features_spatial/
│   ├── pair_features_resnet/
│   ├── predicate_features_semantic/
│   ├── AGP_annotation_5_train.txt
│   ├── AGP_annotation_5_test.txt
│   └── ...
```
### 3. Training
1. Prepare the configuration file with your own paths and hyperparameters (see `Tools/config` for an example).
2. Run training:
   ```bash
   python main.py
   ```
