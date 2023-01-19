import cv2 as cv
import time
def video2images(video_path,save_path):
    vc = cv.VideoCapture(video_path)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print('openerror!')
        rval = False
    start = 0
    step = 30 # 视频帧计数间隔次数
    while rval:
        
        rval, frame = vc.read()
        if start % step == 0:
            cv.imwrite(f'{save_path}/{time.time()}.jpg', frame)
        start += 1
        cv.waitKey(1)
        
        
if __name__ == '__main__':
    video2images('yongchun.mp4','../files/images')