# 🎼 JamendoMaxCaps: A Large-Scale Music-Caption Dataset with Imputed Metadata

<div align="center">
<a href="https://arxiv.org/abs/2502.07461">📄 Paper</a> |
<a href="https://huggingface.co/datasets/AMAAI-Lab/JamendoMaxCaps">🎵 Dataset</a>

<br/>

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/AMAAI-Lab/JamendoMaxCaps) 
[![arXiv](https://img.shields.io/badge/arXiv-2502.07461-brightgreen.svg)](https://arxiv.org/abs/2502.07461)
</div>

## 📌 Overview
JamendoMaxCaps is a large-scale dataset of **200,000+ instrumental tracks** sourced from the Jamendo platform. It includes **generated music captions** and enhanced **imputed metadata**. We also introduce a retrieval system that leverages both musical features and metadata to identify similar songs, which are then used to fill in missing metadata using a local large language model (LLLM). This dataset supports research in **music-language understanding, retrieval, representation learning, and AI-generated music tasks.**

## ✨ Features
✅ **200,000+ Instrumental Tracks** from Jamendo  
✅ **State-of-the-Art Music Captions** generated using a cutting-edge model  
✅ **Metadata Imputation** using a retrieval-enhanced LLM (Llama-2)  
✅ **Comprehensive Musical and Metadata Features**:
   - 🎵 **MERT-based audio embeddings**
   - 📝 **Flan-T5 metadata embeddings**
   - 🔍 **Imputed metadata fields** (genre, tempo, mood, instrumentation)  

---

## ⚡ Installation Guide
```bash
git clone https://github.com/AMAAI-Lab/JamendoMaxCaps.git
cd JamendoMaxCaps
conda create -n jamendomaxcaps python=3.10
pip install -r requirements.txt
```

---

## 🚀 Usage
### 🎼 Extract MERT Features
```bash
python extract_mert.py
```
Ensure input and output folders are correctly configured.

### 📝 Get Metadata Features
```bash
python process_metadata.py
```
Adjust input and output folder paths accordingly.

### 🔍 Build Unified Retrieval System
```bash
python build_retrival_system.py --weight_audio <weight_audio> --weight_metadata <weight_metadata>
```

### 🎶 Find Top Similar Songs
```bash
python retrieve_similar_entries.py --config <config_file_path>
```

### 🛠️ Run Metadata Imputation
```bash
python metadata_imputation.py
```

---

## 📖 Citation
If you use **JamendoMaxCaps**, please cite:
```bibtex
@article{royjamendomaxcaps2025,
  author    = {Abhinaba Roy, Renhang Liu, Tongyu Lu, Dorien Herremans},
  title     = {JamendoMaxCaps: A Large-Scale Music-Caption Dataset with Imputed Metadata},
  year      = {2025},
  journal   = {arXiv:2502.07461}
}
```

---

## 🤝 Acknowledgments
JamendoMaxCaps is built upon **Creative Commons-licensed music** from the Jamendo platform and leverages advanced AI models, including **MERT, Flan-T5, and Llama-2**. Special thanks to the research community for their invaluable contributions to open-source AI development!

---
📜 **[Read the Paper](https://arxiv.org/abs/xxxxx)** | 🎵 **[Explore the Dataset](https://huggingface.co/datasets/AMAAI-Lab/JamendoMaxCaps)**

