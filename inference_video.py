#-*-coding:utf-8-*-
# date:2020-09-23
# Author: Eric.Lee
# function: inference pose video

import os
import cv2
import numpy as np
import torch
import time
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import random
from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

def process_data(img, img_size=416):# 图像预处理
    # img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad
#---------------------------------------------------------

class light_pose_model(object):
    def __init__(self,
        model_path='finetune_model/light_pose.pth',
        heatmaps_thr = 0.05,
        track = 1,
        smooth = 1,
        ):

        self.model_path=model_path
        self.height_size=256

        self.track = track
        self.smooth = smooth

        self.net = PoseEstimationWithMobileNet()

        checkpoint = torch.load(self.model_path, map_location='cpu')

        load_state(self.net, checkpoint)

        self.net = self.net.eval()
        self.net = self.net.cuda()

        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.previous_poses = []
        self.dict_id_color = {}
        self.heatmaps_thr = heatmaps_thr

    def predict(self, img,vis = False):
        with torch.no_grad():
            heatmaps, pafs, scale, pad = infer_fast(self.net, img, self.height_size, self.stride, self.upsample_ratio, False)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(self.num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx],self.heatmaps_thr, all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
            current_poses = []
            Flag_Pose = False
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)
                Flag_Pose = True

            if Flag_Pose == False:
                return None

            if self.track:
                track_poses(self.previous_poses, current_poses, smooth=self.smooth)
                self.previous_poses = current_poses
            dict_id_color_r = {}
            for id_ in self.dict_id_color.keys():
                flag_track = False
                for pose in current_poses:
                    if id_ ==  pose.id:
                        flag_track = True
                        break
                if flag_track:
                    dict_id_color_r[pose.id] = self.dict_id_color[pose.id]
            dict_id_color = dict_id_color_r

            for pose in current_poses:
                if pose.id not in self.dict_id_color.keys():
                    R_ = random.randint(30,255)
                    G_ = random.randint(30,255)
                    B_ = random.randint(30,255)
                    self.dict_id_color[pose.id] = [[B_,G_,R_],1]
                else:
                    self.dict_id_color[pose.id][1] += 1
            pose_dict = {}
            pose_dict['data'] = []
            for pose in current_poses:
                keypoints_list = []
                for k in range(pose.keypoints.shape[0]):
                    keypoints_list.append((float(pose.keypoints[k][0]),float(pose.keypoints[k][1])))

                dict_ = {
                    'bbox': (float(pose.bbox[0]),float(pose.bbox[1]),float(pose.bbox[2]),float(pose.bbox[3])),
                    'id': str(pose.id),
                    'keypoints': keypoints_list,
                    'color': (float(self.dict_id_color[pose.id][0][0]),
                             float(self.dict_id_color[pose.id][0][1]),
                             float(self.dict_id_color[pose.id][0][2])),
                    }
                pose_dict['data'].append(dict_)
            if vis:
                for pose in pose_dict['data']:
                    bbox = pose['bbox']
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 255, 0),3)

                    cv2.putText(img, 'ID: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1]) - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),4)
                    cv2.putText(img, 'ID: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1] - 16)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    draw_one_pose(img,np.array(pose['keypoints']),(int(pose['color'][0]),int(pose['color'][1]),int(pose['color'][2])))

        return pose_dict

def draw_one_pose(img,keypoints,color_x = [255, 0, 0]):

    color = [0, 224, 255]

    for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        global_kpt_a_id = keypoints[kpt_a_id, 0]
        if global_kpt_a_id != -1:
            x_a, y_a = keypoints[kpt_a_id]
            cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        global_kpt_b_id = keypoints[kpt_b_id, 0]
        if global_kpt_b_id != -1:
            x_b, y_b = keypoints[kpt_b_id]
            cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
        if global_kpt_a_id != -1 and global_kpt_b_id != -1:
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255,60,60), 5)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color_x, 2)

if __name__ == '__main__':
    video_path = "./video/demo_05.mp4" # 加载视频
    # video_path = 0 # 加载相机
    model_path = "./pre_train_model/light_pose-20210519.pth"

    model_pose = light_pose_model(model_path = model_path,heatmaps_thr = 0.08) # 定义模型推理类
    print("load:{}".format(model_path))
    video_capture = cv2.VideoCapture(video_path)

    flag_write_video = True # 是否记录推理 demo 视频
    print('flag_write_video',flag_write_video)
    flag_video_start = False
    video_writer = None

    while True:
        ret, im0 = video_capture.read()
        if ret:

            if flag_video_start == False  and flag_write_video:
                loc_time = time.localtime()
                str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
                video_writer = cv2.VideoWriter("./demo/demo_{}.mp4".format(str_time), cv2.VideoWriter_fourcc(*"mp4v"), fps=24, frameSize=(int(im0.shape[1]), int(im0.shape[0])))
                flag_video_start = True

            pose_dict = model_pose.predict(im0.copy())
            if pose_dict is not None:
                for pose in pose_dict['data']:
                    bbox = pose['bbox']
                    # cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])),
                    #               (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (25, 155, 255),2)

                    cv2.putText(im0, 'ID: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1]) - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),9)
                    # cv2.putText(im0, 'ID: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1] - 16)),
                    #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    draw_one_pose(im0,np.array(pose['keypoints']),(int(pose['color'][0]),int(pose['color'][1]),int(pose['color'][2])))

            cv2.namedWindow('image',0)
            cv2.imshow('image',im0)
            if flag_write_video and flag_video_start:
                video_writer.write(im0)

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    if flag_write_video:
        video_writer.release()
