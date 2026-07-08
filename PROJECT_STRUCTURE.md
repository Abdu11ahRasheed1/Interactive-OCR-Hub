```
├── backend/
│   ├── app.py            # FastAPI main router API
│   ├── ocr_engine.py     # Segmentation, Feature extraction, Model training and prediction
│   ├── models/           # Stores scaler.pkl, knn_model.pkl, and cnn_model.keras
│   └── dataset/          # Grid template datasets segmented folders
├── frontend/
│   ├── index.html        # Glassmorphic UI Dashboard page
│   ├── css/
│   │   └── style.css     # Dark mode CSS library
│   └── js/
│       └── app.js        # Canvas drawing and upload listeners
├── requirements.txt      # Project requirements
└── README.md             # Guide documentation
```