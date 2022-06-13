from flask import Flask, request ,send_file, send_from_directory
from flask_cors import CORS ,cross_origin
import os, shutil
from PIL import Image
import io
import cv2
from web import StructureSimilarity

app = Flask(__name__, static_folder="houghtransform/build", static_url_path="")
CORS(app)

PATH='outputs/'
ALLOWED_EXTENSIONS = set(['PNG', 'JPG', 'JPEG'])

STSIM=StructureSimilarity('model_200.pth')
# STSIM=StructureSimilarity('http://s3.amazonaws.com/local-structure-similarity/model_200.pth')

# redirect the landing page
# @app.route('/api',methods=['GET'])
# @cross_origin()
# def get_landing_page():
#     return {"tutorial":"Flask React Heoruk"}


# Revice as a post request the 2 images to proccess and run the model on the. 
# In reponse the main image with the detected objects is retrived.
@app.route("/uploadimage",methods=['POST'])
@cross_origin()
def upload_first_file():
    requestedPicSize=request.args["picsize"]
    dir= PATH + str(abs(hash(request.host)))
    
    if not os.path.isdir(dir):
        os.mkdir(dir)

    uploaded_file = request.files['fileToSearchIn']
    uploaded_sketch = request.files['sketch']
    
    # uploaded_file.save(dir+'/fileToSearchIn.'+str(uploaded_file.mimetype).split('/')[1])
    # uploaded_sketch.save(dir+'/sketch.'+str(uploaded_sketch.mimetype).split('/')[1])
    
    # imgToVerify1=Image.open('./'+dir+'/fileToSearchIn.'+str(uploaded_file.mimetype).split('/')[1])
    # imgToVerify2=Image.open('./'+dir+'/sketch.'+str(uploaded_sketch.mimetype).split('/')[1])

    uploaded_file.save(dir+'/fileToSearchIn')
    uploaded_sketch.save(dir+'/sketch')
    
    imgToVerify1=Image.open('./'+dir+'/fileToSearchIn')
    imgToVerify2=Image.open('./'+dir+'/sketch')

    if((imgToVerify1.format not in ALLOWED_EXTENSIONS) or (imgToVerify2.format not in ALLOWED_EXTENSIONS)):
        return 'invalid request - first go to /',400

    #Use the model to generate the files locateobjects.jpg and heatmap.jpg localy on server.
    tgt_im_rgb=cv2.imread('./'+dir+'/fileToSearchIn',cv2.IMREAD_COLOR)
    ref_im=cv2.imread('./'+dir+'/sketch',cv2.IMREAD_COLOR)
    im , hm =STSIM.locate_ref_in_tgt(tgt_im_rgb,ref_im,64)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    hm=cv2.cvtColor(hm,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./'+dir+'/im.png',im)
    cv2.imwrite('./'+dir+'/hm.png',hm)
    # Model finished

    return send_file(dir+'/im.png',as_attachment=True)
        
# Need to prevent some unknown from trying to get heatmap - he is unknown so we will not have a heatmap for him.    
# revice as a get request for the heat map. this request SHOULD only come after the uploadimage post request. 
# in reponse the hetp map is retrived.
@app.route("/heatmap",methods=['GET'])
@cross_origin()
def upload_second_file():
    dir= PATH + str(abs(hash(request.host)))
    if not os.path.isdir(dir):
        return 'invalid request - first go to /',400

    return send_file(dir+'/hm.png',as_attachment=True)

@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder,'index.html')    

if __name__=='__main__':
    app.run()