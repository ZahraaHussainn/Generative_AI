# 🧠 Generative AI Projects – Zahra Hussain

Welcome to my **Generative AI** repository! This collection brings together key experiments, models, and fine-tuning techniques from my coursework at BDS-8A, under the guidance of **Dr. Hajra Waheed**. Each section highlights a major area of generative modeling or fine-tuning in NLP and Computer Vision.

---

## 📁 Contents

- `Assignment : ViT Anti-Spoofing, CLIP Retrieval, Stable Diffusion`
- `Assignment : Transformer Fine-Tuning with PEFT`
- `Assignment on GANs and on Variational Autoencoders (VAEs)` 
- Reports in `.pdf` or `.docx` format
- Training scripts 

---

## 🧪 Assignment : ViT Anti-Spoofing, CLIP Visual Search & Stable Diffusion

This assignment focuses on three core applications of generative and pretrained vision models:

### 🔐 Face Anti-Spoofing with ViT
- **Model**: Vision Transformer (ViT-B/16)
- **Task**: Classify real vs. spoofed face images from the CelebA-Spoof dataset.
- **Result**: Achieved ~99% accuracy with perfect recall, showing strong real-world reliability for face verification systems.

### 🖼️ CLIP-Based Image Retrieval
- **Model**: `openai/clip-vit-base-patch32`
- **Task**: Retrieve COCO validation images based on text queries using image-text similarity.
- **Highlight**: Demonstrated zero-shot learning by retrieving semantically relevant images for complex text queries.

### 🎨 Artistic Image Generation with Stable Diffusion
- **Model**: `runwayml/stable-diffusion-v1-5`
- **Task**: Image-to-image translation using different text prompts like "pixel art", "Van Gogh", etc.
- **Highlight**: Explored how prompt, strength, guidance, and inference steps affect visual output.

---

## 📘 Assignment 3: Sentiment Classification with Transformer Fine-Tuning

This assignment explored multiple strategies to fine-tune transformers on a subset of the IMDb movie review dataset for binary sentiment classification:

### 🔁 Fine-Tuning Approaches
- **Full Fine-Tuning (RoBERTa)**: High accuracy (88.63%), high compute cost.
- **LoRA (BERT)**: Lightweight, trainable with limited hardware. Accuracy: 62.65%.
- **QLoRA (BERT-Tiny)**: Optimized for low-memory environments. Accuracy: 50%.
- **IA3 (RoBERTa)**: Efficient tuning with high performance (Accuracy: 100%).

Each method highlights trade-offs between performance, memory, and training time using Hugging Face + PEFT libraries.

---

## 🧬 GANs (Generative Adversarial Networks)

*In this section, I explored GAN architectures for image generation or fake image detection.*

- **Concept**: GANs consist of a Generator and Discriminator in adversarial training.
- **Work**: worked on Mnist dataset and generated outputs 

---

## 🧮 Variational Autoencoders (VAEs)

*This section focuses on probabilistic generative models using VAEs.*

- **Concept**: VAEs encode inputs into a latent space with a regularized distribution, then reconstruct samples.
- **Use Case**: Image generation, anomaly detection.


---

## 🛠️ Technologies Used

- Python, PyTorch, Hugging Face Transformers & Datasets
- CLIP, ViT, Stable Diffusion (via Diffusers)
- PEFT Library for LoRA, QLoRA, IA3
- COCO, MNIST & CelebA-Spoof datasets
- Visualization with Matplotlib and image libraries

---

## License

This repository is created for academic and learning purposes. Please cite appropriately if using any component of this work.

