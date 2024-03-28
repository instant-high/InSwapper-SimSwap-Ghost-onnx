import numpy as np
import os
import glob
import sys
import subprocess
import platform

import cv2
import shutil
import imutils

from argparse import ArgumentParser
from tqdm import tqdm

import insightface
from insightface.app import FaceAnalysis

import onnxruntime as rt
rt.set_default_logger_severity(3)


parser = ArgumentParser()
parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
parser.add_argument("--target_video", default='./assets/driving.mp4', help="path to target video")
parser.add_argument("--result_video", default='./result.mp4', help="path to output")

parser.add_argument("--audio", dest="audio", action="store_true", help="Keep audio")

parser.add_argument("--sharpen", default=False, action="store_true", help="Slightly sharpen swapped face")
parser.add_argument("--enhancement", default='none', choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=5, type=float, help='Amount of face enhancement blending 1 - 10')

parser.add_argument("--flip_image", dest="flip_image", action="store_true", help="flip source image vertically") # kind of best match

parser.add_argument("--segmentation", action="store_true", help="Use face_segmentation mask")
parser.add_argument("--swap_index", default="1,2,5", type=lambda x: list(map(int, x.split(','))),help='index of swaped parts')

parser.add_argument("--face_parser", action="store_true", help="Use face_parsing mask")
parser.add_argument("--parser_index", default="1, 2, 3, 4, 5, 6, 10, 11, 12, 13", type=lambda x: list(map(int, x.split(','))),help='index of swaped parts')


parser.add_argument("--face_occluder", action="store_true", help="Use occluder face mask")
parser.add_argument("--no_chin", action="store_true", help="Extra mask to exclude mouth+chin")
parser.add_argument("--scale", default=10, help="Scale factor input video, 10 = 1")
parser.add_argument("--startpos", dest="startpos", type=int, default=0, help="Frame to start from")
parser.add_argument("--endpos", dest="endpos", type=int, default=0, help="Frame to end inference")


opt = parser.parse_args()

angle = 0
btn_down = False

swap_index = opt.swap_index
assert type(swap_index) == list

parser_index = opt.parser_index
assert type(parser_index) == list

from face_swapper.utils import paste_back, align_crop
from face_swapper import ArcFace, Ghost, SimSwap, SimSwapUnofficial

onnx_params = {
    'sess_options': rt.SessionOptions(),
    'providers': ["CUDAExecutionProvider", "CPUExecutionProvider"]
}

#ghost_path = "face_swapper/ghost_unet_1_block.onnx"
ghost_path = "face_swapper/ghost_unet_2_block.onnx"
#ghost_path = "face_swapper/ghost_unet_3_block.onnx"
swapper = Ghost(ghost_path, **onnx_params)
ghost_backbone_path = "face_swapper/ghost_arcface_backbone.onnx"
backbone = ArcFace(ghost_backbone_path, **onnx_params)


app = FaceAnalysis(name='buffalo_l',device="cuda")
app.prepare(ctx_id=0, det_size=(320,320))

if opt.enhancement == 'gpen':
    from enhancers.GPEN.GPEN import GPEN
    gpen256 = GPEN(model_path="enhancers/GPEN/GPEN-BFR-256.onnx", device="cuda")
if opt.enhancement == 'gfpgan':
    from enhancers.GFPGAN.GFPGAN import GFPGAN   
    gfpganv14 = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device="cuda")
if opt.enhancement == 'codeformer':
    from enhancers.Codeformer.Codeformer import CodeFormer
    codeformer = CodeFormer(model_path="enhancers/Codeformer/codeformer.onnx", device="cuda")
if opt.enhancement == 'restoreformer':
    from enhancers.Restoreformer.Restoreformer import RestoreFormer
    restoreformer = RestoreFormer(model_path="enhancers/Restoreformer/restoreformer.onnx", device="cuda")

if opt.segmentation:
    from seg_mask.seg_mask import SEGMENTATION_MODULE
    seg_module = SEGMENTATION_MODULE(model_path="seg_mask/vox-5segments.onnx", device="cuda")
if opt.face_parser:
    from face_parser.face_parser import FACE_PARSER
    facemask = FACE_PARSER(model_path="face_parser/face_parser.onnx", device="cuda")
if opt.face_occluder:
    from face_occluder.face_occluder import FACE_OCCLUDER
    occluder = FACE_OCCLUDER(model_path="face_occluder/face_occluder.onnx", device="cuda")
    
#-------------- Draw line for target video pre-alignment --------------#

def get_points(im):

    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.setMouseCallback("unique_window_identifier", mouse_handler, data)
    cv2.imshow("unique_window_identifier",im)
    cv2.setWindowTitle("unique_window_identifier", "Draw left to right eye - any key to accept")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    points = np.uint16(data['lines'])

    return points, data['im']

def mouse_handler(event, x, y, flags, data):
    global angle,ix,iy,ex,ey,btn_down

    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #if you release the button, finish the line
        btn_down = False
        data['lines'][0].append((x, y)) #append the seconf point
        cv2.circle(data['im'], (x, y), 7, (0, 0, 255),1)
        cv2.line(data['im'], data['lines'][0][0], data['lines'][0][1], (0,0,255), 1)
        cv2.imshow("unique_window_identifier", data['im'])
        ex = x
        ey = y
        dX=(ey-iy)
        dY=(ex-ix)
        angle = int(np.degrees(np.arctan2(dY, dX)) - 90)
        print (angle)

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #thi is just for a line visualization
        image = data['im'].copy()
        cv2.line(image, data['lines'][0][0], (x, y), (0,0,255), 1)
        cv2.imshow("unique_window_identifier", image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 9:   #anzahl versuche
        btn_down = True
        data['lines'].insert(0,[(x, y)]) #prepend the point
        cv2.circle(data['im'], (x, y), 7, (0, 0, 255), 1)
        cv2.imshow("unique_window_identifier", data['im'])
        cv2.setWindowTitle("unique_window_identifier", "Draw left to right eye - any key to accept")
        ix = x
        iy = y

def kps(img):
    detect_results = app.get(img)
    for face in detect_results:
        kps = face.kps
    return kps        
        
def video_swap(source_image, target_video, result_video, enhance_mode):

    print("Running on " + rt.get_device())
    print("")
    assert insightface.__version__>='0.7'
    assert enhance_mode in ['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer']
    
    blend = opt.blending/10
    global angle
        
    # static mask 
    r_mask = np.zeros((256,256), dtype=np.uint8)
    r_mask =  cv2.ellipse(r_mask, (128,128), (90,108),0,0,360,(255,255,255), -1)
    if opt.no_chin:
        r_mask = cv2.rectangle(r_mask,(10,165),(246,256),(0,0,0), -1)
    
    r_mask = cv2.resize(r_mask,(224,224))    
    r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
    
    for i in range(10):
        r_mask = cv2.GaussianBlur(r_mask,(19,19),cv2.BORDER_DEFAULT)
 
    r_mask = r_mask/255

    source_face = cv2.imread(source_image)
                 
    if opt.flip_image:
        source_face = source_face[:, ::-1]
        
    source_face = cv2.imread(opt.source_image)
    if opt.flip_image:
        source_face = source_face[:, ::-1]
 
       
    source_embedding = backbone.forward(source_face, kps(source_face))
    
    cap = cv2.VideoCapture(opt.target_video)
    w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames

    if opt.endpos == 0:
        opt.endpos = total_frames
            
    cutout = opt.endpos - opt.startpos # frames to process from cut in -> cut out
    
    cap.set(1,opt.startpos)
    
    global audio_pos
    audio_pos = (opt.startpos/fps) # for merging original video audiotrack from startpos

    #resize input        
    scale  = int(opt.scale)/10
    w_out = int(w_out*scale)
    h_out = int(h_out*scale)
    ret, cropped_frame = cap.read()
    cropped_frame = cv2.resize(cropped_frame,(w_out,h_out))
    
    
    #crop face region
    print ("")
    print ("Select region of face to replace")
    print ("")

    showCrosshair = False
    show_cropped = cropped_frame
         
    roi = cv2.selectROI("Select region of face to replace", show_cropped,showCrosshair)

    if roi == (0, 0, 0, 0):
        roi = (0,0,w_out,h_out)
        cropped_roi = cropped_frame
    else:
        roiw=roi[2]
        roih=roi[3]
        if roi[2] %2 !=0 : roiw=(roi[2])-1
        if roi[3] %2 !=0 : roih=(roi[3])-1
        roi = (roi[0],roi[1],roiw,roih)			    
        cropped_roi = cropped_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
      
    #print (roi)
    (roi_h, roi_w) = cropped_roi.shape[:2]
    cv2.destroyAllWindows()

    #draw angle
    cv2.imshow("unique_window_identifier", cropped_frame)
    pts, final_image = get_points(cropped_frame)
    cv2.setWindowTitle("unique_window_identifier", "Draw left to right eye - 'Enter' to accept")
    cv2.imshow('unique_window_identifier', final_image)
    cv2.destroyAllWindows()

    #crop output
    ret, full_frame = cap.read()
    full_frame = cv2.resize(full_frame,(w_out,h_out))
    
    print ("")
    print ("Crop final output video")
    print ("")
    
    showCrosshair = False
    show_cropped = full_frame
         
    rv = cv2.selectROI("Crop final output video", show_cropped,showCrosshair)

    if rv == (0, 0, 0, 0):
        (vh, vw) = full_frame.shape[:2]
        rv = (0,0,vw,vh)
        full_frame = full_frame
    else:    
        full_frame = full_frame[int(rv[1]):int(rv[1]+rv[3]), int(rv[0]):int(rv[0]+rv[2])]

    (h_crop, w_crop) = full_frame.shape[:2]

    cv2.destroyAllWindows()
    
        
    #writer:
    if opt.audio:
        output = cv2.VideoWriter(('_temp.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w_crop,h_crop))
    else:
        output = cv2.VideoWriter((opt.result_video),cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w_crop,h_crop))

    os.system('cls')
    cap.set(1,opt.startpos)
    x = 0
    for frame_idx in tqdm(range(cutout)):

        ret, full_frame = cap.read()
        if not ret:
            break
        
        #resize input video    
        full_frame = cv2.resize(full_frame,(w_out,h_out))
        ori_frame = full_frame.copy()
        (h,w) = full_frame.shape[:2]
        
        #crop face region
        face_region = full_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] #+ crop region
        face_region_copy = face_region.copy()
        (hc,wc) = face_region.shape[:2]

        # - rotate pre angle
        face_region = imutils.rotate_bound(face_region, angle) # rotate pre angle
        (h_rot, w_rot) = face_region.shape[:2]

        res = face_region.copy()
            
        try:

            target_cropped, matrix = align_crop(face_region, kps(face_region), swapper.align_crop_size, mode=swapper.align_crop_mode)
            swapped_face = swapper.forward(target_cropped, source_embedding)

            swapped_orig = swapped_face.copy()
            swapped_orig = cv2.resize(swapped_orig,(256,256))
            
            facematrix_rev = cv2.invertAffineTransform(matrix)
            
            #-------------------

            if opt.sharpen:
                smoothed = cv2.GaussianBlur(swapped_face, (9, 9), 10)
                swapped_face = cv2.addWeighted(swapped_face, 1.3, smoothed, -0.3, 0)
                                
            if enhance_mode == 'none':
                swapped_face = cv2.resize(swapped_face,(256,256),interpolation=cv2.INTER_LANCZOS4) ##(256,256),interpolation=cv2.INTER_LANCZOS4)
                             
            if enhance_mode == 'gpen':
                swapped_face = cv2.resize(swapped_face,(256,256))
                swapped_face = gpen256.enhance(swapped_face)
                swapped_face = cv2.addWeighted(swapped_face.astype(np.float32),blend, swapped_orig.astype(np.float32), 1.-blend, 0.0)

            if enhance_mode == 'gfpgan':
                swapped_face = cv2.resize(swapped_face,(512,512))           
                swapped_face = gfpganv14.enhance(swapped_face)
                swapped_face = cv2.resize(swapped_face,(256,256))
                swapped_face = cv2.addWeighted(swapped_face.astype(np.float32),blend, swapped_orig.astype(np.float32), 1.-blend, 0.0)
            
            if enhance_mode == 'codeformer':                
                swapped_face = cv2.resize(swapped_face,(512,512))
                swapped_face = codeformer.enhance(swapped_face,0.6)
                swapped_face = cv2.resize(swapped_face,(256,256))
                swapped_face = cv2.addWeighted(swapped_face.astype(np.float32),blend, swapped_orig.astype(np.float32), 1.-blend, 0.0)
                            
            if enhance_mode == 'restoreformer':
                swapped_face = cv2.resize(swapped_face,(512,512))
                swapped_face = restoreformer.enhance(swapped_face)
                swapped_face = cv2.resize(swapped_face,(256,256))
                swapped_face = cv2.addWeighted(swapped_face.astype(np.float32),blend, swapped_orig.astype(np.float32), 1.-blend, 0.0)

                #------------------
            #faceparsing mask 5seg
            if opt.segmentation:
                r_mask = seg_module.mask(swapped_face, swap_index)
                r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
                r_mask = cv2.rectangle(r_mask, (5,5), (251,251), (0, 0, 0), 10)
                if opt.no_chin:
                    r_mask = cv2.rectangle(r_mask,(10,165),(246,256),(0,0,0), -1)
                r_mask = cv2.resize(r_mask,(224,224))
                r_mask = cv2.GaussianBlur(r_mask,(19,19),cv2.BORDER_DEFAULT)
                r_mask = r_mask /255
                #cv2.imshow("SegMask",r_mask)
                                    
            #faceparsing mask 19parts
            if opt.face_parser:
                r_mask = facemask.create_region_mask(swapped_face, parser_index)
                r_mask = cv2.resize(r_mask,(256,256))
                r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
                r_mask = cv2.rectangle(r_mask, (5,5), (251,251), (0, 0, 0), 10)
                if opt.no_chin:
                    r_mask = cv2.rectangle(r_mask,(10,165),(246,256),(0,0,0), -1)
                r_mask = cv2.resize(r_mask,(224,224))
                r_mask = cv2.GaussianBlur(r_mask,(19,19),cv2.BORDER_DEFAULT)
                #cv2.imshow("Parser",r_mask)
                    
            #occluder mask                
            if opt.face_occluder:
                r_mask = occluder.create_occlusion_mask(swapped_face)
                r_mask = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2RGB)
                r_mask = cv2.rectangle(r_mask, (5,5), (251,251), (0, 0, 0), 10)
                if opt.no_chin:
                    r_mask = cv2.rectangle(r_mask,(10,165),(246,256),(0,0,0), -1)
                r_mask = cv2.resize(r_mask,(224,224))
                r_mask = cv2.GaussianBlur(r_mask,(19,19),cv2.BORDER_DEFAULT)
                #cv2.imshow("Occluder",r_mask)
                                       
   
                                       
            mask = cv2.warpAffine(r_mask, facematrix_rev,(w_rot,h_rot))
            mask = imutils.rotate_bound(mask, angle *-1)
            #print (swapped_face.dtype, face_region_copy.dtype)
            #input (mask.shape)
            cv2.imshow("M",mask)
            
            swapped_face = cv2.resize(swapped_face,(224,224))
            swapped_orig = cv2.resize(swapped_orig,(224,224))    
            swapped_face = cv2.warpAffine(swapped_face, facematrix_rev,(w_rot,h_rot))
            swapped_face = imutils.rotate_bound(swapped_face, angle *-1) # rotate back  pre angle
                
            (vhn,vwn) = swapped_face.shape[:2]
            x1_old = (vwn//2) - (roi_w//2)
            x2_old = (vwn//2) + (roi_w//2)
            y1_old = (vhn//2) - (roi_h//2)
            y2_old = (vhn//2) + (roi_h//2)
            
            swapped_face = swapped_face[y1_old :y2_old , x1_old :x2_old] # alte ausschnitt grosse widerherstellen

            mask = mask[y1_old :y2_old , x1_old :x2_old] # alte ausschnitt grosse widerherstellen            
                                      
            final = (mask * swapped_face + (1-mask) * face_region_copy).astype(np.uint8)
           
            #insert face region
            (vhn,vwn) = final.shape[:2]
            ori_frame[int(roi[1]):int(roi[1])+vhn, int(roi[0]):int(roi[0])+vwn] = final
                            
        except:
            ori_frame = ori_frame
                               
            #crop final output
        final = ori_frame[int(rv[1]):int(rv[1]+rv[3]), int(rv[0]):int(rv[0]+rv[2])]

        cv2.imshow("Final result",final)                        
        k = cv2.waitKey(1)
            
        if k == ord('s'):
            if opt.sharpen == False:
                opt.sharpen = True
            else:
                opt.sharpen = False
            print ('')    
            print ("Sparpen = " + str(opt.sharpen))
                                 
        if k == 27:
            cv2.destroyAllWindows()
            output.release()
            break
                                
        output.write(final)

video_swap(opt.source_image, opt.target_video, opt.result_video, opt.enhancement)

if opt.audio:
    print ("Writing Audio...")
    command = 'ffmpeg.exe -y -vn -ss ' + str(audio_pos) + ' -i ' + '"' + opt.target_video + '"' + ' -an -i ' + '_temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + opt.result_video + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.system('cls')
    #print(command) 
    #input("Face swap done. Press Enter to continue...") 
    if os.path.exists('_temp.mp4'):
        os.remove('_temp.mp4')