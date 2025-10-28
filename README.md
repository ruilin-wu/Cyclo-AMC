

# Cyclo-AMC

This project implements the pipeline of **Cyclo-AMC**.  
It processes raw IQ samples to extract **Spectral Correlation Density (SCD)** features and trains deep neural networks for modulation recognition using the **CSPB.ML.2022** dataset.

---

## 1. SCD Matrix Generation (`CSPBML2022_SCD_generate.ipynb`)

This notebook converts raw time-domain `.tim` IQ data into normalized **SCD matrices** that capture cyclostationary signatures unique to each modulation scheme.

### Main Processing Steps
1. **Data Loading**  
   Reads `.tim` files containing 32,768-sample complex IQ frames, representing modulated signals under various SNR conditions and modulation types.

2. **Segmentation and Windowing**  
   Each frame is segmented into smaller overlapping segments (typically 512 samples) and multiplied by **Hamming** window to reduce spectral leakage.

3. **Spectral Correlation Estimation**  
   Computes the **Spectral Correlation Density (SCD)** using the **Frequency-Shifted Averaged Multiplier (FAM)** algorithm.  
   This step captures periodic correlations caused by modulation-induced cyclostationarity.

4. **Normalization and Downsampling**  
   The generated SCD surfaces are logarithmically scaled, magnitude-normalized, and resized to compact **64×64** matrices suitable for neural network input.

5. **Dataset Export**  
   The resulting tensors are stored as `.pt` files, grouped by modulation type and SNR, ready for model training.

---

## 2. Model Training and Evaluation (`CSPBML_Resnet.ipynb`)

This notebook trains a **ResNet-based classifier** on the generated SCD matrices for supervised modulation classification.

### Training Workflow
1. **Dataset Loading**  
   Loads precomputed SCD matrices and corresponding labels into PyTorch datasets.  
   Each input is a 2-D feature map representing the spectral–cyclic structure of a signal.

2. **Model Architecture**  
   Utilizes a lightweight **ResNet** variant (ResNet18-style) with:
   - 2D convolutional layers (3×3 kernels)  
   - Batch normalization and ReLU activations  
   - Global average pooling and fully connected classification head  
   The model is optimized for both training speed and generalization across SNRs.

3. **Optimization Strategy**  
   - Optimizer: SGD with momentum  
   - Learning rate scheduling (e.g., cosine decay)  
   - Cross-entropy loss with label smoothing to handle class imbalance  
   - Optional dropout or weight decay for regularization

4. **Evaluation Metrics**  
   After each epoch, the notebook reports:
   - Overall classification accuracy  
   - Training loss

5. **Model Saving and Visualization**  
   Trained checkpoints and metrics are saved for further testing or deployment.  
   


