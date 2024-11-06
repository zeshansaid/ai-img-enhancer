from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import base64
from PIL import Image, ImageFile
from enhancer_engine import RealESRGAN
import io
 
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- index ------------------------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------
@app.route('/')
def hello_world():
    return "api , online"


# -----------------------------------------------------------------------------------------------

#------------------------------------ Object removal ---------------------------------------------

 

#-------------------------- Enhancer ------------------------------------------------------


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

Enhance_model = RealESRGAN(device, scale=4)
Enhance_model.load_weights('weights/RealESRGAN_x4.pth', download=True)

@app.route('/upscaler', methods=['POST'])
def enhance_image():
    # Get the uploaded image from the request
    print("/upscaler new request coming")
    data = request.get_json()
    base64Image= data["image"]
    #print(base64Image[:20])
    base64_data = ""
    #extracting the base64 string part 
    comma_index = base64Image.find(',')
    if comma_index != -1:
       base64_string = base64Image[comma_index + 1:]
       base64_data = base64_string
    else:
       print("Comma not found in the data URI.")  
       base64_data = base64Image
    img =  stringToImage(base64_data)
    img.save("upscaler_requsted_image.png")
    new_image = img#Image.open("upscaler_requsted_image.png")
    # Perform image enhancement using the model
    sr_image = Enhance_model.predict(new_image)
    sr_image.save("upscaler_result_image.png")
    bio = io.BytesIO()
    sr_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()
    #print(f"sr_image string = {im_b64[20:]}")
    return jsonify({"bg_image":im_b64})




 
if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False)
    app.run(host="0.0.0.0", port=5000, debug=True,use_reloader=False)
