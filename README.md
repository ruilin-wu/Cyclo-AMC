# Versal-based Implementation of Cyclo-AMC

By default, this project targets the `xcve2802-vsvh1760-2MP-e-S` platform for VEK280.

## Dependencies

This project requires the following:

- **VEK280 Evaluation Kit**
- **Vitis 2024.2**

## Guide

### 1. Downloading the CSPB.ML.2022 Dataset

Navigate to the `DNN` directory:

```bash
cd DNN
````

Download the **CSPB.ML.2022 dataset** from the official website:
[https://cyclostationary.blog/data-sets/](https://cyclostationary.blog/data-sets/)

Store the downloaded `.tim` files in the `CSPB_ML_2022_Data` folder.

---

### 2. Generate the SCD Matrix

```bash
cd DNN
```

Run the notebook:

```bash
CSPBML2022_SCD_generate.ipynb
```

This script converts `.tim` files into 64×64 SCD matrices.

---

### 3. Train the DNN Using the CSPB.ML.2022 Dataset

```bash
cd DNN/network/studnet
```

Run the training notebook:

```bash
CSPBML_Resnet.ipynb
```

This notebook trains a ResNet-based DNN on the preprocessed dataset.

---

### 4. Make the Project for Cyclo-AMC

```bash
cd CycloAMC_AIEML
make all
```

This command builds the full AIE-ML project for Cyclo-AMC.

---

### 5. Check the Mapping to the AIE-ML Array

```bash
cd CycloAMC_AIEML
make summary
```

This command outputs mapping summaries for the AIE-ML array.

