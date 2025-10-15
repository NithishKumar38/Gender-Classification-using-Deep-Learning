# Gender Detection using Deep Learning & OpenCV  

This project is a **deep learning-based gender detection system**.  
It consists of:  
- **Training script** – builds and trains a CNN model on a gender dataset.  
- **Detection script** – uses the trained model to perform real-time gender classification via webcam.  

---
<img width="1920" height="1080" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/e7a8dabf-3f4a-451c-9ef3-09dfd5950a96" />


## 🚀 Features  
- CNN model built with **TensorFlow & Keras**  
- Data augmentation with **ImageDataGenerator**  
- Real-time detection using **OpenCV & cvlib**  
- Tracks and displays **Men/Women count** on live video  
- Training metrics visualization (**loss/accuracy curves**)  

---

## 📂 Project Structure  
.
├── train_model.py # Training script

├── gender_detection.py # Real-time detection script

├── gender_detection1.keras # Saved trained model

├── plot1.png # Training history graph

├── requirements.txt # Dependencies

└── README.md # Documentation


---

## 🛠️ Installation & Setup  

### 1. Clone the Repository  

git clone https://github.com/your-username/gender-detection.git
cd gender-detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📊 Training the Model
Dataset Structure
Copy
Edit
gender_dataset_face/

├── man/

│   ├── face_1.jpg

│   ├── face_2.jpg

├── woman/

│   ├── face_1.jpg

│   ├── face_2.jpg

Run Training

python train_model.py
✔️ Trains a CNN for 100 epochs (configurable)

✔️ Saves trained model as gender_detection1.keras

✔️ Saves training plot as plot1.png

▶️ Real-Time Detection

Run the detection script:

python gender_detection.py

Opens webcam

Detects faces and classifies them as man or woman

Displays bounding boxes with confidence %

Shows live count:

Men: 1, Women: 2

Press q to exit

📦 Requirements
tensorflow

opencv-python

numpy

cvlib

scikit-learn

matplotlib

Install manually:

pip install tensorflow opencv-python numpy cvlib scikit-learn matplotlib

📸 Example Outputs

Training graph (plot1.png): Loss vs Accuracy

Real-time detection: Bounding boxes with labels

🔮 Future Improvements
Expand dataset for higher accuracy

Add age detection alongside gender

Deploy as a Flask/Streamlit web app

🙌 Author
Developed by Mariyappan M 🚀
