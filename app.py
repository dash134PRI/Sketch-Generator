import os
from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def check_opencv_modules():
    # Check for required OpenCV modules
    has_xphoto = hasattr(cv2, 'xphoto')
    return has_xphoto

def convert_to_sketch(image, style='classic'):
    has_xphoto = check_opencv_modules()
    
    # Common preprocessing
    if style != 'high-contrast':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
    
    # Style-specific processing
    if style == 'detailed':
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Post-processing based on style
    if style == 'colored':
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        sketch = cv2.addWeighted(sketch_bgr, 0.5, image, 0.5, 0)
    elif style == 'high-contrast':
        edges = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)
        sketch = cv2.bitwise_not(edges)
    elif style == 'oil-painting':
        if not has_xphoto:
            raise ValueError("Oil painting effect not available. Please install opencv-contrib-python")
        try:
            sketch = cv2.xphoto.oilPainting(image, 7, 1)
        except Exception as e:
            print(f"Oil painting error: {str(e)}")
            raise
    elif style == 'detail-enhanced':
        sketch = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)
    elif style == 'thermal':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sketch = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif style == 'denoised':
        sketch = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif style == 'emboss':
        kernel = np.array([[-2,-1,0], 
                        [-1,1,1],
                        [0,1,2]])
        sketch = cv2.filter2D(image, -1, kernel)
    
    return sketch

@app.route('/')
def home():
    # Check available modules and print status
    has_xphoto = check_opencv_modules()
    print(f"OpenCV modules status - xphoto: {has_xphoto}")
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    style = request.form.get('style', 'classic')
    
    if file.filename == '':
        return 'No file selected', 400
    
    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 'Invalid image file', 400
            
        # Process image
        sketch = convert_to_sketch(image, style=style)
        
        # Convert to bytes
        _, img_encoded = cv2.imencode('.jpg', sketch)
        response = io.BytesIO(img_encoded.tobytes())
        
        return send_file(
            response,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'sketch_{style}.jpg'
        )
    
    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
