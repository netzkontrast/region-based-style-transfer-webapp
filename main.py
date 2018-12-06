import os
from flask import Flask, jsonify, request, redirect, url_for, flash, send_file, jsonify, send_from_directory
from flask import render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from combine import load_frozenmodel, segmentation_image, blend_images, style_transfer, color_transfer
import numpy as np
import cv2

app = Flask(__name__)
app.debug = True
app.secret_key = "super secret key"
app.config.from_object(__name__)
CORS(app)

UPLOAD_PATH = './_upload_images/'
REGION_BASED_STYLE_TRANSFER_PATH = './_region_based_style_transfer_images/'
REGION_BASED_STYLE_TRANSFER_WITH_COLOR_PATH = './_region_based_style_transfer_with_color_images/'
GLOBAL_STYLE_TRANSFER_PATH = './_global_style_transfer_images/'
SAMPLE_PATH = './_sample_images/'
COLOR_TRANSFER_PATH = "./_color_transfer/"
STYLE_PATH = './_style_images/'
STYLE_CPKT_PATH = "./models/style_models"
SEGMENT_MASK_PATH = "./_segment_mask_images/"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def region_based_style_transfer(image_name, image_suffix, style):
    fg_img_path = os.path.join(UPLOAD_PATH, image_name+'.'+image_suffix)
    bg_img_path = os.path.join(GLOBAL_STYLE_TRANSFER_PATH, image_name+'_'+style+'.'+image_suffix)
    style_img_path = os.path.join(STYLE_PATH, style+'.'+image_suffix)
    style_data_path = os.path.join(COLOR_TRANSFER_PATH, "_style_data/" + style+'.txt')
    color_img_path = os.path.join(COLOR_TRANSFER_PATH, "_global_color_transfer_images/" + image_name+"_" +style+"."+image_suffix)
    background_style_transfer_img_path =  os.path.join(REGION_BASED_STYLE_TRANSFER_PATH,  "blend_"+image_name+"_" +style+"."+image_suffix)
    background_style_transfer_with_color_img_path =  os.path.join(REGION_BASED_STYLE_TRANSFER_WITH_COLOR_PATH,  "blend_"+image_name+"_" +style+"_color."+image_suffix)

    graph = load_frozenmodel()
    bin_mask = segmentation_image(graph, image_path=fg_img_path, segmentation_save_path="%s/%s.jpg"%(SEGMENT_MASK_PATH, image_name))

    style_transfer(style, fg_img_path, bg_img_path)
    color_transfer(fg_img_path, style_img_path, style_data_path, color_img_path)

    background_style_blend_img = blend_images(fg_path=fg_img_path, bg_path=bg_img_path, mask=bin_mask)

    background_style_blend_with_color_img = blend_images(fg_path=color_img_path, bg_path=bg_img_path, mask=bin_mask)



    cv2.imwrite(background_style_transfer_img_path, background_style_blend_img)
    cv2.imwrite(background_style_transfer_with_color_img_path, background_style_blend_with_color_img)

    return True

@app.route('/index')
def index():
    return  '''
    <!doctype html>
    <title>COMS4731 Region-based Style Transfer</title>
    <h1>Hello</h1>
    It works!
    '''

@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/<style_cpkt>', methods=['GET', 'POST'])
def upload_file(style_cpkt):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_PATH, filename))

            img_path = filename.rsplit('.', 1)[0]
            img_suffix = filename.rsplit('.', 1)[1].lower()
            style = style_cpkt

            blend_img_name = 'blend_'+img_path+'_'+style+'.'+img_suffix
            blend_img_path = os.path.join(REGION_BASED_STYLE_TRANSFER_PATH, blend_img_name)

            if(not (os.path.isfile(blend_img_path))):
                region_based_style_transfer(img_path, img_suffix, style)

            return redirect(url_for('region_based_style_transfer_image', image_name=blend_img_name))
    return '''
    <!doctype html>
    <title>COMS4731 Region-based Style Transfer</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image>
         <input type=submit value=Upload> 
    </form>
    '''

@app.route('/get_region_based_style_transfer_image/<image_name>', methods=["GET"])
def region_based_style_transfer_image(image_name):
    return send_from_directory(REGION_BASED_STYLE_TRANSFER_PATH, image_name)

@app.route('/get_region_based_style_transfer_with_color_image/<image_name>', methods=["GET"])
def region_based_style_transfer_with_color_image(image_name):
    return send_from_directory(REGION_BASED_STYLE_TRANSFER_WITH_COLOR_PATH, image_name)

@app.route('/get_global_style_transfer_image/<image_name>', methods=["GET"])
def global_style_transfer_image(image_name):
    return send_from_directory(GLOBAL_STYLE_TRANSFER_PATH, image_name)

@app.route('/get_sample_image/<sample_file_name>', methods=["GET"])
def sample_image(sample_file_name):
    return send_from_directory(SAMPLE_PATH, sample_file_name)

@app.route('/get_style_image/<style_file_name>', methods=["GET"])
def style_image(style_file_name):
    return send_from_directory(STYLE_PATH, style_file_name)

@app.route('/get_upload_image/<upload_image_name>', methods=["GET"])
def upload_image(upload_image_name):
    return send_from_directory(UPLOAD_PATH, upload_image_name)

@app.route('/upload', methods=["POST"])
def upload():
    if 'image' not in request.files:
        flash('No file part')
        return jsonify({"error":"no image file"})
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return jsonify({"error":"no selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_PATH, filename))
        return jsonify({"error": "no error"})
    else:
        flash('file format is not allowed')
        return jsonify({"error":"file format is not allowed"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)