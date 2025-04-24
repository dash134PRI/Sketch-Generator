# Sketch Generator

A web application that transforms images into sketches and effects using advanced computer vision techniques. Built with Python, OpenCV, and Flask, this application offers multiple artistic styles and real-time preview capabilities.

## Features

- Intuitive web interface
- Drag-and-drop or click-to-upload functionality
- Real-time preview of both original and processed images
- Multiple styles and effects
- One-click download option
- Responsive design for all screen sizes

## Available Styles

### Classic Effects
- **Classic Pencil**: Traditional black and white pencil sketch effect
- **Detailed Sketch**: Enhanced sketch with better edge definition and sharpness
- **High Contrast**: Bold, dramatic sketch with emphasized light and dark areas
- **Colored Pencil**: Artistic sketch that preserves original color information

### Artistic Effects
- **Oil Painting**: Simulates oil painting texture and brush strokes
- **Detail Enhanced**: Intelligently sharpens and enhances image details
- **Emboss**: Creates a 3D-like relief effect with directional lighting

### Special Effects
- **Thermal Vision**: Heat-map visualization using color gradients
- **Denoised**: Advanced noise reduction while preserving image details

## Requirements

- Python 3.7+
- Flask
- OpenCV-contrib-python
- NumPy
- Pillow (Python Imaging Library)
- Werkzeug

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <path>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:####`

3. Using the interface:
   - Upload an image using the "Choose Image" button or drag and drop
   - Select effect from the style dropdown menu
   - Click "Convert to Sketch" to process the image
   - Preview the result in the output window
   - Click "Download Sketch" to save the processed image

## Technical Details

The application uses following OpenCV techniques for image processing:

### Core Processing
- Grayscale conversion
- Edge detection
- Gaussian blur
- Color space manipulation

### Advanced Techniques
- Adaptive thresholding for high-contrast effects
- Non-local means denoising
- Detail enhancement filters
- Color mapping for special effects
- Custom kernels for emboss and sharpening effects

## Troubleshooting

If you encounter any issues:

1. Common Problems:
   - Make sure all dependencies are correctly installed
   - Check that you're using a supported image format (JPEG, PNG, etc.)
   - Ensure your virtual environment is activated

2. Image Processing Issues:
   - Large images may take longer to process
   - Some effects work better with high-contrast images
   - Ensure adequate lighting in original images

## Contributing

Contributions are welcome! Feel free to:
- Submit bug reports
- Propose new features or styles
- Submit pull requests
- Suggest improvements to documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV for providing powerful image processing capabilities
- Flask for the web framework
- The Python community for various supporting libraries 
