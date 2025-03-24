import io
import base64
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, send_file
from docx import Document
from rembg import remove
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    """Landing page showing tool options."""
    return render_template('index.html')

@app.route('/speech_to_text')
def speech_to_text():
    """Speech-to-text page."""
    return render_template('speech_to_text.html')

@app.route('/download', methods=['POST'])
def download():
    """
    Receives text from the speech-to-text page,
    creates a Word doc, and returns it as a download.
    """
    text = request.form.get('text', '')
    doc = Document()
    doc.add_paragraph(text)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)

    return send_file(
        file_stream,
        as_attachment=True,
        download_name="transcription.docx",
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

@app.route('/remove_background')
def remove_background():
    """Background removal page."""
    return render_template('remove_background.html')

@app.route('/remove_bg', methods=['POST'])
def remove_bg():
    """
    Receives an uploaded image, uses rembg to remove background,
    returns the resulting PNG.
    """
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    input_image = Image.open(file).convert("RGBA")
    output_image = remove(input_image)

    output_stream = io.BytesIO()
    output_image.save(output_stream, format="PNG")
    output_stream.seek(0)

    return send_file(
        output_stream,
        as_attachment=True,
        download_name="no_bg.png",
        mimetype="image/png"
    )

@app.route('/sketch')
def sketch():
    """Sketching tool page."""
    return render_template('sketch.html')

@app.route('/sketch_image', methods=['POST'])
def sketch_image():
    """
    Receives an image file, converts it to a pencil sketch,
    and returns the resulting PNG image.
    """
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    # Read image file as numpy array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Invalid image", 400

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_img = cv2.bitwise_not(gray_img)

    # Blur the inverted image using Gaussian blur
    blurred = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)

    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred)

    # Create the pencil sketch image by dividing the grayscale image by the inverted blur image
    sketch_img = cv2.divide(gray_img, inverted_blur, scale=256.0)

    # Encode the processed image to PNG
    success, buffer = cv2.imencode('.png', sketch_img)
    if not success:
        return "Image processing failed", 500

    output_stream = io.BytesIO(buffer)
    output_stream.seek(0)

    return send_file(
        output_stream,
        as_attachment=True,
        download_name="sketch.png",
        mimetype="image/png"
    )

# ===== Excel Merge Tool Endpoints =====

@app.route('/excel_tool', methods=['GET', 'POST'])
def excel_tool():
    """
    Step 1: Upload two Excel files.
    On GET: display a form to upload File A and File B.
    On POST: read the files, determine common columns, and then
    show a new form with merge options.
    """
    if request.method == 'GET':
        return render_template('excel_tool.html')
    else:
        file_a = request.files.get('file_a')
        file_b = request.files.get('file_b')
        if not file_a or not file_b:
            return "Both files are required", 400

        # Read file bytes
        file_a_data = file_a.read()
        file_b_data = file_b.read()
        
        # Read Excel files with pandas
        df_a = pd.read_excel(io.BytesIO(file_a_data))
        df_b = pd.read_excel(io.BytesIO(file_b_data))
        
        cols_a = list(df_a.columns)
        cols_b = list(df_b.columns)
        common_cols = list(set(cols_a).intersection(set(cols_b)))
        
        # Encode file data in base64 to pass via hidden fields
        file_a_b64 = base64.b64encode(file_a_data).decode('utf-8')
        file_b_b64 = base64.b64encode(file_b_data).decode('utf-8')
        
        return render_template('excel_tool_options.html',
                               common_cols=common_cols,
                               cols_a=cols_a,
                               cols_b=cols_b,
                               file_a_b64=file_a_b64,
                               file_b_b64=file_b_b64)

@app.route('/process_excel', methods=['POST'])
def process_excel():
    """
    Process the Excel merge based on user selections.
    Expected form fields:
      - common_col: the common column for merging
      - copy_from: 'A' or 'B' indicating which file to copy extra columns from
      - selected_cols: list of columns (from the copy_from file) to copy
      - file_a_b64, file_b_b64: hidden fields containing base64-encoded file data
    The result is an Excel file merging the base file (the one not chosen as copy-from)
    with the selected columns from the chosen file.
    """
    common_col = request.form.get('common_col')
    copy_from = request.form.get('copy_from')
    selected_cols = request.form.getlist('selected_cols')
    file_a_b64 = request.form.get('file_a_b64')
    file_b_b64 = request.form.get('file_b_b64')
    
    if not all([common_col, copy_from, file_a_b64, file_b_b64]):
        return "Missing data", 400
    
    file_a_data = base64.b64decode(file_a_b64)
    file_b_data = base64.b64decode(file_b_b64)
    
    df_a = pd.read_excel(io.BytesIO(file_a_data))
    df_b = pd.read_excel(io.BytesIO(file_b_data))
    
    # Determine which dataframe to copy from and which to use as base
    if copy_from == 'A':
        copy_df = df_a
        base_df = df_b
    else:
        copy_df = df_b
        base_df = df_a
    
    # Create a subset of the copy dataframe containing the common column and the selected columns
    copy_subset = copy_df[[common_col] + selected_cols]
    
    # Merge the base dataframe with the copy subset on the common column (left join)
    result = base_df.merge(copy_subset, on=common_col, how='left')
    
    # Write the result to an Excel file in memory
    output_stream = io.BytesIO()
    result.to_excel(output_stream, index=False)
    output_stream.seek(0)
    
    return send_file(
        output_stream,
        as_attachment=True,
        download_name="merged.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == '__main__':
    app.run(debug=True)
