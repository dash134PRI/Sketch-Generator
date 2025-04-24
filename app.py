import os
from flask import Flask, request, render_template, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename
from functools import wraps
import time

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_IMAGE_SIZE = (2000, 2000)  # Maximum image dimensions
RATE_LIMIT = {'calls': 0, 'reset_time': time.time(), 'limit': 100}  # 100 calls per minute

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Reset counter every minute
        if time.time() - RATE_LIMIT['reset_time'] > 60:
            RATE_LIMIT['calls'] = 0
            RATE_LIMIT['reset_time'] = time.time()
        
        RATE_LIMIT['calls'] += 1
        if RATE_LIMIT['calls'] > RATE_LIMIT['limit']:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        
        return f(*args, **kwargs)
    return decorated_function

def check_opencv_modules():
    # Check for required OpenCV modules
    has_xphoto = hasattr(cv2, 'xphoto')
    return has_xphoto

def resize_if_needed(image):
    height, width = image.shape[:2]
    if height > MAX_IMAGE_SIZE[1] or width > MAX_IMAGE_SIZE[0]:
        # Calculate new dimensions while maintaining aspect ratio
        ratio = min(MAX_IMAGE_SIZE[0]/width, MAX_IMAGE_SIZE[1]/height)
        new_size = (int(width * ratio), int(height * ratio))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def convert_to_sketch(image, style='classic'):
    try:
        # Resize image if needed
        image = resize_if_needed(image)
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
                raise ValueError(f"Oil painting error: {str(e)}")
        elif style == 'detail-enhanced':
            sketch = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)
        elif style == 'thermal':
            # Convert to grayscale and normalize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize the image to enhance contrast
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            # Apply thermal colormap
            sketch = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            # Enhance contrast of the final result
            sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=10)
        elif style == 'denoised':
            sketch = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif style == 'emboss':
            kernel = np.array([[-2,-1,0], 
                            [-1,1,1],
                            [0,1,2]])
            sketch = cv2.filter2D(image, -1, kernel)
        
        return sketch
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/')
def home():
    # Check available modules and print status
    has_xphoto = check_opencv_modules()
    print(f"OpenCV modules status - xphoto: {has_xphoto}")
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
@rate_limit
def convert_image():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        style = request.form.get('style', 'classic')
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Read and validate image
        img_bytes = file.read()
        if len(img_bytes) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 16MB'}), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        sketch = convert_to_sketch(image, style=style)
        
        # Convert to bytes
        _, img_encoded = cv2.imencode('.jpg', sketch, [cv2.IMWRITE_JPEG_QUALITY, 90])
        response = io.BytesIO(img_encoded.tobytes())
        
        return send_file(
            response,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'sketch_{style}.jpg'
        )
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(429)
def too_many_requests(e):
    return jsonify({'error': 'Too many requests. Please try again later.'}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    # Use environment variables for host and port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
