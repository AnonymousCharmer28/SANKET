# S.A.N.K.E.T.

### Sign Analysis & Neural Keypoint Extraction Technology

S.A.N.K.E.T. is a computer vision and machine learning system designed to recognize **Indian Sign Language (ISL) fingerspelling** and convert recognized hand gestures into text.

The project explores a landmark-based approach to sign recognition, using hand keypoints extracted from images rather than relying directly on raw image pixels.

---

## 🪷 Project Overview

### The Problem

Communication barriers can make everyday interactions difficult for people who use sign language, particularly when the people around them do not understand it.

Indian Sign Language has its own linguistic structure and alphabetic representations. Building an accessible recognition system for ISL requires handling variations in hand position, orientation, lighting, background, and the use of one or both hands.

### The SANKET Approach

SANKET follows a **computer vision → feature extraction → machine learning** pipeline.

Instead of feeding complete images directly into a traditional image-classification model, the system extracts hand landmarks and represents each gesture as numerical spatial features.

These features are then used to train a machine learning classifier to recognize the corresponding ISL alphabet.

### High-Level Pipeline

```text
                  S.A.N.K.E.T. PIPELINE

        ┌───────────────────────┐
        │      Input Images     │
        │       ISL Dataset     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Hand Landmark       │
        │      Extraction       │
        │      MediaPipe        │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Numerical Feature   │
        │     Representation    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Machine Learning    │
        │       Training        │
        │   Random Forest       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Trained Model      │
        │   + Label Encoder     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Gesture → ISL Letter  │
        │       → Text          │
        └───────────────────────┘
```

---

## ✨ Key Features

* **Two-Handed Spatial Tracking** — Supports simultaneous tracking of up to two hands and captures their corresponding landmark coordinates for gesture representation.
* **Hand Landmark Extraction** — Converts hand images into structured landmark coordinates.
* **Gesture-Based Representation** — Uses spatial hand-keypoint information as machine learning features.
* **ISL Alphabet Recognition** — Designed around recognition of Indian Sign Language fingerspelling.
* **Machine Learning Classification** — Uses a Random Forest classifier for gesture classification.
* **Feature-Based Pipeline** — Separates computer vision feature extraction from model training.
* **Portable Model Artifacts** — Trained models can be serialized for later integration with an application or API.
* **Text-to-Speech (TTS) Integration:** Features native browser-based vocalization to convert compiled ISL text strings into audible speech, closing the communication loop.
* **Accessibility-First Roadmap:** Transitioning from manual interface controls to custom hand-gesture triggers for completely hands-free interaction.

---

## 🛠️ Technology Stack

| Technology       | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| **Python**       | Core development language                       |
| **MediaPipe**    | Hand landmark detection and keypoint extraction |
| **OpenCV**       | Image processing and image input                |
| **Pandas**       | Dataset and feature-data handling               |
| **Scikit-learn** | Machine learning and model training             |
| **Joblib**       | Model serialization                             |

---

## 📁 Project Structure

```text
SANKET/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
│
├── models/
│   ├── rf_model.pkl
│   └── label_encoder.pkl
│
├── extract.py
├── train.py
├── isl_features.csv
├── .gitignore
└── README.md
```

> **Note:** The dataset, generated feature files, and trained model artifacts should remain excluded from the repository unless their respective licenses and project requirements permit redistribution.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SANKET
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install mediapipe opencv-python pandas scikit-learn joblib
```

---

## 📊 Dataset Preparation

Create a `dataset/` directory in the project root and organize images according to their corresponding class labels.

Example:

```text
dataset/
├── A/
├── B/
├── C/
├── D/
└── ...
```

The dataset itself should **not be committed to this repository** unless redistribution is explicitly permitted.

---

## 🔬 Feature Extraction

The feature extraction stage processes the dataset and detects hand landmarks using MediaPipe.

The system is configured to detect up to **two hands simultaneously**, allowing landmark information from both hands to be represented within the extracted feature set.

Run:

```bash
python extract.py
```

The extracted landmark information is converted into numerical features and stored in the feature dataset used for subsequent model training.

---

## 🤖 Model Training

After feature extraction, the generated feature data is used to train the classification model.

Run:

```bash
python train.py
```

The training pipeline:

1. Loads the extracted feature data.
2. Separates features and class labels.
3. Encodes the categorical labels.
4. Splits the data for training and validation.
5. Trains the Random Forest classifier.
6. Evaluates the model.
7. Saves the trained model and label encoder.

The resulting model artifacts are stored in the `models/` directory.

---

## 🧠 Why Landmark-Based Recognition?

Raw images contain a large amount of information that may not be directly relevant to recognizing a hand gesture.

Landmark extraction provides a more structured representation of the hand by reducing the visual input to key spatial points.

This creates a pipeline where:

```text
Image
  ↓
Hand Detection
  ↓
Landmark Extraction
  ↓
Numerical Features
  ↓
Machine Learning
  ↓
Gesture Classification
```

The approach provides a lightweight foundation that can later be extended toward real-time recognition and application-level integration.

---

## 🚀 Future Scope

Potential future development includes:

* Real-time webcam-based recognition
* Continuous gesture recognition
* Text composition from sequential signs
* Confidence scoring
* Web-based user interface
* Backend API integration
* Model optimization and benchmarking
* Support for a broader range of ISL vocabulary
* Hands-free gesture-driven TTS activation
* End-to-end accessibility workflow optimization

---

## ⚠️ Current Limitations

The current system is focused on **fingerspelling recognition** rather than complete Indian Sign Language translation.

Recognition performance may also depend on factors such as:

* Dataset quality and diversity
* Hand positioning
* Lighting conditions
* Occlusion
* Camera quality
* Variations between signers

Further evaluation is required before making claims about real-world or production-level performance.

---

## 📌 Project Status

**Currently under development.**

S.A.N.K.E.T. is being developed as part of the **MGMCET Tech Community Cohort**.

---

## 📚 Dataset Acknowledgement

The training data utilizes an Indian Sign Language image dataset compiled by researchers at the **Chandigarh College of Engineering and Technology, Panjab University, University Institute of Engineering and Technology**. It is utilized here strictly for research and academic development purposes.

source: https://data.mendeley.com/datasets/7tsw22y96w/1

---

## 👩‍💻 Contributors

Developed as a cohort project by the SANKET team.

---

## 📄 License

License information will be added after the project and dataset usage requirements are finalized.
