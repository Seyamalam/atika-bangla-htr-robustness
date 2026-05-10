**Objective:** Evaluate the **Cross-Dataset Robustness** of state-of-the-art Bangla Handwritten Text Recognition (HTR) models.

# **1\. Main Problem**

Handwritten Bangla text recognition is still difficult because Bangla handwriting contains:

* different writing styles,  
* connected characters,  
* মাত্রা,  
* যুক্তাক্ষর,  
* broken strokes,  
* irregular spacing,  
* page skew and noise.

Existing Bangla handwritten OCR datasets such as **BN-HTRd** are useful because they contain labelled handwritten Bangla document images. However, models trained on public datasets may not work well on real-world handwritten images.

This dataset of **2703 handwritten Bangla images** can be used to study this real-world gap.

---

# **2\. Main Research Idea**

The thesis will investigate:

How well do Bangla handwritten text recognition models trained on a labelled public dataset perform on real-world handwritten Bangla images?

In simple words:

Train/test model on BN-HTRd  
↓  
Then test or analyze performance on your own real-world handwritten images  
↓  
Find where and why the model fails

# 

# 

# **3\. Datasets**

## **Dataset 1: BN-HTRd**

This is the **main labelled dataset**.

Will use it for:

* model training,  
* validation,  
* testing,  
* CER/WER evaluation.

  BN-HTRd already has handwritten Bangla document images with ground-truth labels, so it solves the annotation problem.

  ## **Dataset 2: My Own Dataset**

  2703 handwritten Bangla images

  Use this as a **real-world external dataset**.

  Because the dataset is not fully labelled, we can use it in two ways:

  ### **Use 1: Dataset quality and preprocessing analysis**

  You can analyze:

* image noise,  
* skew,  
* lighting,  
* handwriting variation,  
* page layout,  
* ID/watermark removal,  
* line spacing,  
* background artifacts.

  ### **Use 2: Small annotated subset**

  We will select:

  50–100 images

  or better:

  300–500 line images

  Then manually annotate only those selected samples.

  This small subset can be used as an **external test set**.

# **5\. Recommended Workflow**

## **Phase 1: Dataset Preparation**

### **BN-HTRd**

Download BN-HTRd and organize it like this:

BN\_HTRd/  
├── images/  
├── line\_images/  
├── labels.csv  
├── train.csv  
├── val.csv  
└── test.csv

Use line-level data first.

That means:

handwritten line image → Bangla line text

Do not start with full-page OCR immediately.

### **Self Dataset**

Keep this structure:

bangla\_ocr\_dataset/  
├── Raw images/  
├── processed\_images/  
├── line\_images/  
├── labels/  
└── splits/

This dataset will be used for external analysis and limited annotation.

**Phase 2: Image Preprocessing**

Apply preprocessing to both BN-HTRd and own images.

Methods:

grayscale conversion  
noise removal  
contrast enhancement  
binarization  
deskewing  
cropping  
resizing  
line normalization

Purpose:

To see whether preprocessing improves handwritten Bangla OCR performance.

## **Phase 3: Model Selection**

Use existing models first. 

|  | Model | Purpose |  |
| ----- | ----- | ----- | :---- |
| **Traditional** | Tesseract OCR | traditional baseline |  |
| **Standard DL** | CRNN \+ CTC | standard handwritten text recognition baseline |  |
| **Modern DL** | CNN \+ BiLSTM \+ Attention | stronger sequence model | Add an Attention layer to the sequence model |
| **SOTA** | Transformer-based HTR | advanced model | **TrOCR** or **GraDeT-HTR**. *Implementation Note:* Fine-tune microsoft/trocr-base-handwritten. For Bangla, you **must** use a **Grapheme-based Tokenizer** because standard Unicode tokenization fails on complex **যুক্তাক্ষর** (compound characters).  |

## **Phase 4: Training and Testing**

### **Experiment 1: BN-HTRd baseline**

Train/test models on BN-HTRd.

BN-HTRd train set → train model  
BN-HTRd test set → evaluate model

Metrics:

CER: Character Error Rate  
WER: Word Error Rate

### **Experiment 2: Preprocessing comparison**

Compare performance before and after preprocessing.

Raw BN-HTRd images → OCR model → CER/WER  
Processed BN-HTRd images → OCR model → CER/WER

This will show whether preprocessing improves recognition.

### **Experiment 3: External dataset testing**

Use the own annotated subset.

Model trained on BN-HTRd  
↓  
Test on your own handwritten Bangla images  
↓  
Measure performance drop

### 

### **Key Resources** 

* **Primary Dataset (BN-HTRd):** [Download/Repo](https://github.com/crusnic-corp/BN-DRISHTI)  
  * Use the BN-HTRd\_Splitted version on HuggingFace for line-level training.  
* **SOTA Model (GraDeT-HTR 2025/26):** [ResearchGate Paper](https://www.researchgate.net/publication/397423385_GraDeT-HTR_A_Resource-Efficient_Bengali_Handwritten_Text_Recognition_System_utilizing_Grapheme-based_Tokenizer_and_Decoder-only_Transformer)  
  * This is the new 2026 standard for Bangla. It uses a **Decoder-only Transformer**.  
* **TrOCR Fine-Tuning Guide:** [HuggingFace/Colab Tutorial](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb)

