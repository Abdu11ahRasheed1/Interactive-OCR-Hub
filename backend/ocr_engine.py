import os
import cv2
import pickle
import numpy as np
import base64
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu

# Optional tensorflow imports to fall back gracefully if not configured
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    HAS_TF = True
except ImportError:
    pass

class OCREngine:
    def __init__(self, dataset_dir="c:/Users/chras/Desktop/IMP/Projects/backend/dataset", models_dir="c:/Users/chras/Desktop/IMP/Projects/backend/models"):
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Standard alphabet mapping (1-indexed matching lines 651-652 of OCR.ipynb)
        self.char_map = {
            'a': 1, 'd': 2, 'f': 3, 'h': 4, 'k': 5, 'm': 6, 'n': 7,
            'o': 8, 'p': 9, 'q': 10, 'r': 11, 's': 12, 'u': 13,
            'w': 14, 'x': 15, 'z': 16
        }
        self.char_map_rev = {v: k for k, v in self.char_map.items()}

        # 0-indexed alphabetical mapping for Keras CNN (alphabetic folder ordering)
        self.cnn_classes = sorted(list(self.char_map.keys())) # ['a', 'd', 'f', ...]
        self.char_map_rev_cnn = {i: c for i, c in enumerate(self.cnn_classes)}

        self.knn_path = os.path.join(self.models_dir, "knn_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.cnn_path = os.path.join(self.models_dir, "cnn_model.keras")

    def segment_characters(self, image_data, is_bytes=False, min_size=25, padding=2):
        """
        Segment character blobs from main grid template.
        """
        if is_bytes:
            arr = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_data, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            return []

        # Threshold to binary (Otsu plus inversion)
        _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Connected component labelling - contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours left-to-right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_boxes_and_contours = sorted(zip(bounding_boxes, contours), key=lambda x: x[0][0])

        segmented = []
        for i, (bbox, contour) in enumerate(sorted_boxes_and_contours):
            x, y, w, h = bbox
            if w * h < min_size or w < 3 or h < 3:
                continue

            # Extract character patch
            roi = thresh[y:y+h, x:x+w]
            
            # Apply padding
            roi_padded = cv2.copyMakeBorder(roi, padding, padding, padding, padding, 
                                           cv2.BORDER_CONSTANT, value=0)
            
            # Resize
            roi_28 = cv2.resize(roi_padded, (28, 28), interpolation=cv2.INTER_AREA)

            segmented.append({
                "index": i + 1,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "roi_28": roi_28,
                "roi_padded": roi_padded
            })
        return segmented

    def extract_hu_moments(self, roi_binary):
        """
        Extract Hu Moments from binary character region
        """
        # skimage moments need foreground=1 (0-1 range)
        roi_scaled = roi_binary / 255.0
        m = moments(roi_scaled)
        if m[0, 0] == 0:
            return np.zeros(7)
        
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi_scaled, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        return hu

    def store_training_template(self, label_char, image_bytes):
        """
        Segment grid template of a specific character and upload components directory.
        """
        if label_char not in self.char_map:
            raise ValueError(f"Label '{label_char}' is not in allowed class mapping.")

        char_folder = os.path.join(self.dataset_dir, "character_model_single", label_char)
        os.makedirs(char_folder, exist_ok=True)

        segmented = self.segment_characters(image_bytes, is_bytes=True)
        count = 0
        for item in segmented:
            roi_28 = item["roi_28"]
            count += 1
            filename = f"{label_char}_{count}_{np.random.randint(1000, 9999)}.bmp"
            save_path = os.path.join(char_folder, filename)
            cv2.imwrite(save_path, roi_28)
        
        return count

    def train_models(self):
        """
        Trains both K-NN and CNN models using segmented template directory files.
        """
        data_dir = os.path.join(self.dataset_dir, "character_model_single")
        if not os.path.exists(data_dir):
            return {"success": False, "message": "Training folder not constructed yet."}

        train_features = []
        train_labels = []
        class_counts = {}

        folders = os.listdir(data_dir)
        for char_folder in folders:
            if char_folder not in self.char_map:
                continue
            folder_path = os.path.join(data_dir, char_folder)
            label_id = self.char_map[char_folder]
            
            bmp_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")]
            class_counts[char_folder] = len(bmp_files)

            for file in bmp_files:
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                hu = self.extract_hu_moments(img)
                train_features.append(hu)
                train_labels.append(label_id)

        if len(train_features) == 0:
            return {"success": False, "message": "No training characters found. Segment grid template first."}

        # --- 1. Train K-NN Classifier ---
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier

        scaler = StandardScaler()
        X_train = scaler.fit_transform(np.array(train_features))
        y_train = np.array(train_labels)

        knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
        knn.fit(X_train, y_train)

        # Save KNN model and standard scaler
        with open(self.knn_path, "wb") as f:
            pickle.dump(knn, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # --- 2. Train CNN using Keras ---
        cnn_success = False
        cnn_message = "Tensorflow/Keras not configured on local environment."

        if HAS_TF:
            try:
                # Image dataset generation parameters
                img_size = (32, 32)
                batch_size = min(16, len(train_features))
                
                # Check validation availability (need at least 2 items per class or split gracefully)
                train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
                
                train_data = train_gen.flow_from_directory(
                    data_dir,
                    target_size=img_size,
                    color_mode='grayscale',
                    class_mode='sparse',
                    batch_size=batch_size,
                    subset='training',
                    shuffle=True
                )
                
                val_data = train_gen.flow_from_directory(
                    data_dir,
                    target_size=img_size,
                    color_mode='grayscale',
                    class_mode='sparse',
                    batch_size=batch_size,
                    subset='validation',
                    shuffle=False
                )

                # Exactly user requested Sequential architecture layers
                cnn_model = models.Sequential([
                    layers.Input(shape=(32, 32, 1)),
                    layers.Conv2D(32, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D(2, 2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(16, activation='softmax')  # 16 classes
                ])

                cnn_model.compile(
                    optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']
                )
                
                # Fit CNN model
                epochs = 10 if len(train_features) > 100 else 4
                cnn_model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=0)
                
                # Save CNN model weights
                cnn_model.save(self.cnn_path)
                cnn_success = True
                cnn_message = "CNN trained successfully."
            except Exception as e:
                cnn_message = f"Error during CNN training: {str(e)}"
        
        return {
            "success": True,
            "knn_success": True,
            "cnn_success": cnn_success,
            "cnn_message": cnn_message,
            "class_counts": class_counts
        }

    def predict_image(self, image_bytes, method="knn"):
        """
        Segment and predict characters using either KNN or CNN.
        """
        segmented = self.segment_characters(image_bytes, is_bytes=True)
        if len(segmented) == 0:
            return {"text": "", "predictions": [], "success": True}

        predictions = []
        predicted_text_list = []

        if method == "knn":
            if not os.path.exists(self.knn_path) or not os.path.exists(self.scaler_path):
                return {"error": "K-NN Model not trained yet."}
            
            with open(self.knn_path, "rb") as f:
                knn = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                scaler = pickle.load(f)

            for char_info in segmented:
                roi_28 = char_info["roi_28"]
                hu = self.extract_hu_moments(roi_28)
                feats = scaler.transform([hu])
                pred_id = knn.predict(feats)[0]
                char = self.char_map_rev.get(pred_id, "?")
                
                try:
                    distances, _ = knn.kneighbors(feats, n_neighbors=1)
                    prob = float(1.0 / (1.0 + distances[0][0]))
                except:
                    prob = 1.0

                _, buffer = cv2.imencode('.png', roi_28)
                base64_str = base64.b64encode(buffer).decode('utf-8')

                predicted_text_list.append(char)
                predictions.append({
                    "index": char_info["index"],
                    "bbox": char_info["bbox"],
                    "label": char,
                    "prob": float(prob),
                    "roi_image": f"data:image/png;base64,{base64_str}",
                    "hu_moments": [float(val) for val in hu]
                })

        elif method == "cnn":
            if not HAS_TF:
                return {"error": "Tensorflow/Keras is not configured on this server environment."}
            if not os.path.exists(self.cnn_path):
                return {"error": "CNN Model not trained yet."}
            
            cnn_model = models.load_model(self.cnn_path)
            for char_info in segmented:
                roi_padded = char_info["roi_padded"]
                roi_32 = cv2.resize(roi_padded, (32, 32), interpolation=cv2.INTER_AREA)
                roi_norm = roi_32 / 255.0
                roi_input = np.expand_dims(np.expand_dims(roi_norm, axis=-1), axis=0) # (1, 32, 32, 1)

                pred = cnn_model.predict(roi_input, verbose=0)
                pred_id = np.argmax(pred, axis=1)[0]
                char = self.char_map_rev_cnn.get(pred_id, "?")
                prob = float(pred[0][pred_id])

                _, buffer = cv2.imencode('.png', roi_padded)
                base64_str = base64.b64encode(buffer).decode('utf-8')

                predicted_text_list.append(char)
                predictions.append({
                    "index": char_info["index"],
                    "bbox": char_info["bbox"],
                    "label": char,
                    "prob": prob,
                    "roi_image": f"data:image/png;base64,{base64_str}",
                    "hu_moments": []
                })

        return {
            "success": True,
            "text": "".join(predicted_text_list),
            "predictions": predictions
        }

    def get_status(self):
        """
        Check training status and details.
        """
        knn_trained = os.path.exists(self.knn_path) and os.path.exists(self.scaler_path)
        cnn_trained = os.path.exists(self.cnn_path)
        
        data_dir = os.path.join(self.dataset_dir, "character_model_single")
        class_counts = {}
        if os.path.exists(data_dir):
            for folder in os.listdir(data_dir):
                if folder in self.char_map:
                    fd = os.path.join(data_dir, folder)
                    class_counts[folder] = len([f for f in os.listdir(fd) if f.lower().endswith(".bmp")])

        return {
            "knn_trained": knn_trained,
            "cnn_trained": cnn_trained,
            "has_tf": HAS_TF,
            "class_counts": class_counts,
            "allowed_classes": list(self.char_map.keys())
        }
