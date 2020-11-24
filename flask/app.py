from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import sys

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = './static/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['JPEG', 'JPG', 'PNG', 'GIF']


def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            if image.enhanced_filename == "":
                print("No enhanced_filename")
                return redirect(request.url)
            if allowed_image(image.enhanced_filename):
                filename = secure_filename(image.enhanced_filename)
                image.save(os.path.join(app.config['IMAGE_UPLOADS'], filename))
                return render_template('index.html', image_org=filename)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
