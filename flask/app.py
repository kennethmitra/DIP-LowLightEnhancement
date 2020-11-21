from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)


images = UploadSet('images', IMAGES)

app.config['UPLOADED_IMAGES_DEST'] = '/uploads'
configure_uploads(app, (images,))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        filename = images.save(request.files['image'])
        return filename
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
