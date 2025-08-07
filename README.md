# TopoGSL: A Topology-Aware Generative Self-Supervised Learning Framework for Molecular Property Prediction

This repository contains the official implementation of the paper:  
**"TopoGSL: A Topology-Aware Generative Self-Supervised Learning Framework for Molecular Property Prediction."**

## Running Experiments

- **Fine-tuning on a pre-trained TopoGSL model**  
  Run the script:
  ```bash
  ./runExperiment.sh
  ```

- **Full-stage pre-training and fine-tuning**  
  Run the script:
  ```bash
  ./runDistributionExperiment.sh
  ```

## Configuration

All experiment configurations can be found and modified in:

```
Arguments.py
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chasecheng529/TopoGSL.git
   cd TopoGSL
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n TopoGSL python=3.9
   conda activate TopoGSL
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

> All required Python packages are listed in `requirements.txt`.  
