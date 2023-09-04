import cv2
import os,sys
#from PIL import Image
import numpy as np
import argparse
from multiprocessing import Pool

def ToImg(raw_flow,bound):
    ## Split the optical image and normalize the optical image to 0-255
    flow_x=raw_flow[...,0]
    flow_x[flow_x > bound] = bound
    flow_x[flow_x < -bound] = -bound
    flow_x -= -bound
    flow_x *= (255 / float(2 * bound))
    #picture_x=Image.fromarray(flow_x)

    flow_y = raw_flow[..., 1]
    flow_y[flow_y>bound]=bound
    flow_y[flow_y<-bound]=-bound
    flow_y-=-bound
    flow_y*=(255/float(2*bound))
    #picture_y= Image.fromarray(flow_y)
    return flow_x,flow_y
    #return picture_x,picture_y

def flow2img(flow, BGR=True):
	x, y = flow[..., 0], flow[..., 1]
	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
	ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
	hsv[..., 0] = (an / 2).astype(np.uint8)
	hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
	hsv[..., 2] = 255
	if BGR:
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	else:
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img

def parse_args():
    parser = argparse.ArgumentParser(description=
                                     "calculate flow using opencv in different ways")
    parser.add_argument('--video_path',default="./../data/Avenue/Train/"
                        ,type=str,help="the path of videos that need to calcualte")
    parser.add_argument('--dst_path', default="./../data/Avenue/flow-train/",
                        type=str,help="generate flow image path")
    parser.add_argument('--bound', default=15,
                        type=int, help='set the maximum of optical flow')
    parser.add_argument('--multi_Progress', default=3,
                        type=int, help='Number of threads opened')
    args = parser.parse_args()
    return args

def main():
    global args
    args=parse_args()
    # print(args.video_path)
    # print(args.dst_path)
    # print(args.bound)
    # print(args.multi_Progress)

    jpegs_path = args.dst_path + 'jpegs'
    diffs_path = args.dst_path + 'diffs'
    flows_path = args.dst_path + 'flows'
    if not os.path.exists(jpegs_path):
        os.mkdir(jpegs_path)
    if not os.path.exists(diffs_path):
        os.mkdir(diffs_path)
    if not os.path.exists(flows_path):
        os.mkdir(flows_path)
        os.mkdir(flows_path+'/'+'u')
        os.mkdir(flows_path + '/'+'v')


    video_names=os.listdir(args.video_path)
    for video_name in video_names:
        cap = cv2.VideoCapture(args.video_path + video_name)
        if cap.isOpened():
            frames = cap.get(7)-1  ## the frames of videos
            print(video_name+":",frames,"frames")
            tmp=video_name.split('.')[0]
            dst_jpegs_path=jpegs_path +'/'+ tmp
            dst_diffs_path=diffs_path + '/'+tmp
            dst_flows_u_path=flows_path+'/u/' + tmp
            dst_flows_v_path = flows_path +'/v/' + tmp
            if not os.path.exists(dst_jpegs_path):
                os.mkdir(dst_jpegs_path)
            if not os.path.exists(dst_diffs_path):
                os.mkdir(dst_diffs_path)
            if not os.path.exists(dst_flows_u_path):
                os.mkdir(dst_flows_u_path)
            if not os.path.exists(dst_flows_v_path):
                os.mkdir(dst_flows_v_path)

            ret, src_1 = cap.read()
            if(ret==False):
                raise BaseException("1:get video frames error,please check the videos\n")
            for i in range(1, int(frames)):
                ret, src_2 = cap.read()
                if (ret == False):
                    raise BaseException("2:get video frames error,please check the videos\n")
                save_img(dst_path=dst_jpegs_path+'/',sn=i,src=src_2)   ###sn=the series numer in videos
                gray_1 = cv2.cvtColor(src_1, cv2.COLOR_BGR2GRAY)
                gray_2 = cv2.cvtColor(src_2, cv2.COLOR_BGR2GRAY)
                save_diff(dst_path=dst_diffs_path+'/',sn=i,src1=gray_1,src2=gray_2)
                ###src1:background image;src2=current image
                save_flow(dst_path_u=dst_flows_u_path+'/',dst_path_v=dst_flows_v_path+'/',sn=i,src1=gray_1,src2=gray_2,bound=args.bound)  ##optical flow
                ###src1:previous image;src2=current image
                src_1=src_2
        else:
            raise BaseException("can't open videos,please check the path or opencv\n")


def save_img(dst_path,sn,src):
    picture_name='img_' + str(sn).zfill(5) + '.jpg'##eg.picture_neme=img_00001.jpg
    cv2.imwrite(dst_path+picture_name,src)

def save_diff(dst_path,sn,src1,src2):
    picture_name = 'diff_' + str(sn).zfill(5) + '.jpg'  ##eg.picture_neme=diff_00001.jpg

    diff = cv2.absdiff(src1, src2)###absdiff::::src1=background image;src2=current image
    cv2.imwrite(dst_path + picture_name,diff)

def save_flow(dst_path_u,dst_path_v,sn,src1,src2,bound):
    picture_name = 'flow_' + str(sn).zfill(5) + '.jpg'  ##eg.picture_neme=diff_00001.jpg

    #method 1,dense flow,paper:Two-Frame Motion Estimation Based on PolynomialExpansion
    #flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    #method 2,paper:
    #flow = cv2.optflow.calcOpticalFlowSparseToDense(gray1, gray2, grid_step=5, sigma=0.5)

    # method 3,dense,paper:A Duality Based Approach for Realtime TV-L1 Optical Flow.
    dtvl1 = cv2.createOptFlow_DualTVL1()
    flow = dtvl1.calc(src1, src2, None)
    #flow=cv2.createOptFlow_DualTVL1(gray1, gray2, None)

    # method 4,dense flow,paper:SimpleFlow: A Non-iterative, Sublinear Optical FlowAlgorithm,2012
    #flow = cv2.optflow.calcOpticalFlowSF(src1, src2, 3, 5, 5)

    flow_x,flow_y=ToImg(flow, bound)
    cv2.imwrite(dst_path_u+picture_name,flow_x)
    cv2.imwrite(dst_path_v+picture_name,flow_y)

    #flow_bgr=flow2img(flow, BGR=True) #you can use the function to save bgr format.

if __name__ =='__main__':
    main()