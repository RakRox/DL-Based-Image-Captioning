# 🖼️ Image Caption Generator

A **Deep Learning–based Image Captioning System** using the **BLIP (Salesforce/blip-image-captioning-base)** model. This project generates captions for any input image using a pretrained vision–language transformer.

This repository includes:

* Backend caption generation code
* A Gradio-based user interface
* Sample output images
* Complete step-by-step guide for Kaggle and local setups

---

## 🚀 Features

* Caption generation using BLIP
* Accepts **image upload** and **image URL**
* Runs smoothly on **Kaggle GPU (T4)**
* Simple and clean Gradio UI
* Beginner-friendly structure

---

## 📂 Project Structure

```
Image-Caption-Generator/
│
├── backend_captioning_code.py       # Backend model + caption generation code
├── gradio_interface.py              # Gradio UI for captioning
├── ImageCaption.ipynb               # Jupyter Notebook version
├── requirements.txt                 # Python dependencies
│
├── output_samples/                  # Folder containing sample output images
│   ├── 1.PNG
│   ├── 2.PNG
│   └── .gitkeep
│
├── .gitattributes                   # Git settings
└── README.md                        # Project documentation
```

---

## 🛠️ Requirements

Your `requirements.txt` must include:

```
transformers
torch
torchvision
pillow
gradi o
requests
```

---

## 📥 Running the Project on Kaggle

Follow this process to execute your captioning model.

### **1️⃣ Create a Kaggle Notebook**

* Go to **Kaggle → Notebooks**
* Click **New Notebook**
* Enable **GPU (T4)** under accelerator

---

### **2️⃣ Upload Your Input Image**

* Left sidebar → **Add Data** → **Upload** → select your image
* Kaggle assigns a path like:

```
/kaggle/input/yourfoldername/your_image.jpg
```

---

### **3️⃣ Update Image Path in Code**

Inside `backend_captioning_code.py`, edit:

```python
caption = predict_caption("/kaggle/input/yourfoldername/your_image.jpg")
```

Replace:

* `yourfoldername` → uploaded dataset folder
* `your_image.jpg` → your actual filename

**Example:**

```python
caption = predict_caption("/kaggle/input/dogphoto/dog.png")
```

---

### **4️⃣ Run the Backend Code**

The BLIP model loads and generates a caption.

---

## 🖥️ Running the Gradio Interface

* Scroll to the Gradio section in the notebook
* Run the cell
* A shareable link will appear

You can:

* Upload an image
* Paste URL
* Get captions instantly

---

## 🧪 Sample Output Images

Below are the provided test images.

### 🖼️ Output 1

![Output 1](output_samples/1.PNG)

### 🖼️ Output 2

![Output 2](output_samples/2.PNG)

---

## 🧩 How the Backend Works

### **1. Load Model**

```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### **2. Define Caption Function**

```python
def predict_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
```

### **3. Generate Caption**

```python
caption = predict_caption("/kaggle/input/yourfolder/image.jpg")
print(caption)
```

---

## 📌 Notes

* Dataset not required — **one image is enough**
* Only update image path
* Works offline after model download

---

## 📞 Contact

Created by **RakRox**.
Open an issue or submit a PR for improvements.

---

⭐ **If you found this useful, please give it a star on GitHub!**
