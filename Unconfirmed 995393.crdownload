import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

print("🐕 DOG BREED IDENTIFICATION - PROFESSIONAL EDITION")
print("=" * 60)

# ========================================
# 1. PATH CONFIGURATION (EASY TO EDIT)
# ========================================
BASE_DIR = r"C:\Users\HP\Desktop\dog breed identification project\archive (2)\images\Images"
MODEL_PATH = "dog_breed_model.keras"  # Modern .keras format
TEST_IMAGE = r"C:\Users\HP\Desktop\dog breed identification project\test_dog.jpg"

def check_paths():
    """Check if paths exist"""
    print(f"📁 Dataset folder: {BASE_DIR}")
    print(f"💾 Model file: {MODEL_PATH}")
    
    if os.path.exists(BASE_DIR):
        print("✅ Dataset found!")
    else:
        print("⚠️  Dataset path needs updating")
        
    if os.path.exists(MODEL_PATH):
        print("✅ Trained model found!")
    else:
        print("ℹ️  Model will be trained...")
        
    print()

check_paths()

# ========================================
# 2. DATA LOADING & PREPROCESSING
# ========================================
def load_data():
    """Load and prepare training/validation data"""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_data = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    print(f"✅ Loaded {train_data.samples} training images")
    print(f"✅ Loaded {val_data.samples} validation images")
    print(f"✅ Found {len(train_data.class_indices)} dog breeds")
    
    return train_data, val_data

# ========================================
# 3. MODEL BUILDING
# ========================================
def build_model():
    """Build the MobileNetV2 transfer learning model"""
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(120, activation='softmax')  # 120 breeds
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ========================================
# 4. MAIN EXECUTION - SMART TRAIN/LOAD
# ========================================
def run_project():
    """Main function - trains OR loads model intelligently"""
    
    # Check if trained model exists
    if os.path.exists(MODEL_PATH):
        print("⚡ LOADING PRE-TRAINED MODEL (2 seconds)...")
        global model, train_data
        model = load_model(MODEL_PATH)
        train_data, _ = load_data()  # Only for class labels
        print("✅ Model loaded successfully!")
        return True  # Loaded successfully
    
    else:
        print("🎓 TRAINING NEW MODEL (15-30 minutes)...")
        train_data, val_data = load_data()
        
        # Build model
        model = build_model()
        print("\n📊 MODEL ARCHITECTURE:")
        model.summary()
        
        # Train!
        print("\n🚀 STARTING TRAINING...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            verbose=1
        )
        
        # Save model
        model.save(MODEL_PATH)
        print("✅ MODEL SAVED!")
        
        # Plot results
        plot_training_history(history)
        show_project_stats(history)
        
        return True

# ========================================
# 5. VISUALIZATION
# ========================================
def plot_training_history(history):
    """Plot training accuracy curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'bo-', label='Training')
    plt.plot(history.history['val_accuracy'], 'ro-', label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'bo-', label='Training')
    plt.plot(history.history['val_loss'], 'ro-', label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def show_project_stats(history):
    """Display final project statistics"""
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n🎉 PROJECT COMPLETE!")
    print("=" * 50)
    print(f"📁 Model saved: {MODEL_PATH}")
    print(f"🐕 Total breeds: {len(train_data.class_indices)}")
    print(f"📏 Image size: 224x224")
    print(f"📈 Training accuracy: {final_acc*100:.1f}%")
    print(f"✅ Validation accuracy: {final_val_acc*100:.1f}%")
    print("=" * 50)

# ========================================
# 6. PREDICTION FUNCTIONS
# ========================================
def predict_dog_breed(img_path):
    """Predict breed from single image file"""
    if not os.path.exists(img_path):
        return "❌ Error: Image file not found!"
    
    # Load and preprocess
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    confidence = np.max(predictions) * 100
    
    # Get breed name
    class_indices = train_data.class_indices
    labels = {v: k for k, v in class_indices.items()}
    breed_raw = labels[np.argmax(predictions)]
    breed_name = " ".join(breed_raw.split("-")[1:]).title()
    
    if confidence < 50:
        return f"🤔 Not confident ({confidence:.1f}%). Might not be one of our 120 breeds."
    return f"🐶 {breed_name}\n📊 Confidence: {confidence:.1f}%"

def predict_gradio(img):
    """Gradio prediction function"""
    if img is None:
        return "Please upload an image!"
    
    # Convert PIL/numpy to proper format
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array, verbose=0)
    confidence = np.max(predictions) * 100
    
    class_indices = train_data.class_indices
    labels = {v: k for k, v in class_indices.items()}
    breed_raw = labels[np.argmax(predictions)]
    breed_name = " ".join(breed_raw.split("-")[1:]).title()
    
    if confidence < 50:
        return f"⚠️ Low confidence ({confidence:.1f}%). Not one of the 120 breeds?"
    return f"🐕 **{breed_name}**\n📊 Confidence: {confidence:.1f}%"

# ========================================
# 7. GRADIO WEB INTERFACE
# ========================================
def launch_web_app():
    """Launch professional Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# 🐕 **Dog Breed Classifier**")
        gr.Markdown("Powered by MobileNetV2 • 120 breeds • Instant predictions")
        
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Upload dog photo")
                btn = gr.Button("🔍 Identify Breed", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output = gr.Textbox(label="Prediction", lines=3)
        
        # Examples (optional - add sample dog images)
        gr.Examples(
            examples=[],
            inputs=img_input,
            label="Try these examples (add paths to your dog images)"
        )
        
        btn.click(predict_gradio, inputs=img_input, outputs=output)
    
    print("\n🌐 LAUNCHING WEB APP...")
    demo.launch(share=True, show_api=True)

# ========================================
# 8. RUN EVERYTHING!
# ========================================
if __name__ == "__main__":
    success = run_project()
    
    if success:
        print("\n🧪 QUICK TEST:")
        print(predict_dog_breed(TEST_IMAGE))
        
        print("\n🎯 LAUNCH WEB APP? (y/n): ", end="")
        choice = input().lower()
        if choice == 'y':
            launch_web_app()
