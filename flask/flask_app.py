import os
from flask import Flask, render_template, request, redirect, url_for
from image_cap import get_captions

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello(caption=None):
    return render_template('hello.html', caption=caption)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    
    caption= get_captions(f)
    #caption = "test_caption"
    #return redirect(url_for('hello', caption=caption))
    return render_template('hello.html', caption=caption)
