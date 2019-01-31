import os
import os.path as osp
import scipy.io as sio
import numpy as np
from config import cfg
from utils.pose_utils import pixel2cam, rigid_align, process_world_coordinate, warp_coord_to_original
import cv2
import random
from utils.vis import vis_keypoints, vis_3d_skeleton

class Human36M:
    def __init__(self, data_split):
        self.data_split = data_split
        self.data_dir = osp.join('..', 'data', 'Human36M', 'data')
        self.subsampling = self.get_subsampling_ratio(data_split)
        self.joint_num = 18
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        self.flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.lr_skeleton = ( ((8,11),(8,14)), ((11,12),(14,15)), ((12,13),(15,16)), ((0,1),(0,4)), ((1,2),(4,5)), ((2,3),(5,6)) )
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.joints_have_depth = True

        self.action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.subaction_idx = (1, 2)
        self.camera_idx = (1, 2, 3, 4)
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
       
        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
    
    def get_subsampling_ratio(self, data_split):

        if data_split == 'train':
            return 5
        elif data_split == 'test':
            return 200 #64
        else:
            assert 0, print('Unknown subset')

    def load_h36m_annot_file(self, annot_file):
        
        data = sio.loadmat(annot_file)
        joint_world = data['pose3d_world'] # 3D world coordinates of keypoints
        R = data['R'] # extrinsic
        T = np.reshape(data['T'],(3)) # extrinsic
        f = np.reshape(data['f'],(-1)) # focal legnth
        c = np.reshape(data['c'],(-1)) # principal points
        img_heights = np.reshape(data['img_height'],(-1))
        img_widths = np.reshape(data['img_width'],(-1))
       
        # add thorax
        thorax = (joint_world[:, self.lshoulder_idx, :] + joint_world[:, self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((thorax.shape[0], 1, thorax.shape[1]))
        joint_world = np.concatenate((joint_world, thorax), axis=1)

        return joint_world, R, T, f, c, img_widths, img_heights

    def _H36FolderName(self, subject_id, act_id, subact_id, camera_id):
        return "s_%02d_act_%02d_subact_%02d_ca_%02d" % \
               (subject_id, act_id, subact_id, camera_id)

    def _H36ImageName(self, folder_name, frame_id):
        return "%s_%06d.jpg" % (folder_name, frame_id + 1)

    def _AllHuman36Folders(self, subject_list):
        folders = []
        for i in subject_list:
            for j in self.action_idx:
                for m in self.subaction_idx:
                    for n in self.camera_idx:
                        folders.append(self._H36FolderName(i, j, m, n))
        return folders

    def _sample_dataset(self, data_split):
        if data_split == 'train':
            folders = self._AllHuman36Folders([1, 5, 6, 7, 8])
        elif data_split == 'test':
            folders = self._AllHuman36Folders([9, 11])
        else:
            print("Unknown subset")
            assert 0

        return folders

    def load_data(self):

        folders = self._sample_dataset(self.data_split)
        data = []
        for folder in folders:
            
            if folder == 's_11_act_02_subact_02_ca_01':
                continue

            folder_dir = osp.join(self.data_dir, folder)
            
            # load ground truth
            joint_world, R, T, f, c, img_widths, img_heights = self.load_h36m_annot_file(osp.join(folder_dir, 'h36m_meta.mat'))
            img_num = np.shape(joint_world)[0]

            for n in range(0, img_num, self.subsampling):
                
                img_path = osp.join(folder_dir, self._H36ImageName(folder, n))
                joint_img, joint_cam, joint_vis, center_cam, bbox = process_world_coordinate(joint_world[n], self.root_idx, self.joint_num, R, T, f, c)
                
                img_width = img_widths[n]
                img_height = img_heights[n]
                
                data.append({
                    'img_path': img_path,
                    'bbox': bbox, 
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'joint_vis': joint_vis,
                    'center_cam': center_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c
                })

        return data

    def evaluate(self, preds, result_dir):

        print() 
        print('Evaluation start...')

        gts = self.load_data()

        assert len(gts) == len(preds)

        sample_num = len(gts)
        joint_num = self.joint_num
        
        p1_error = np.zeros((sample_num, joint_num, 3)) # PA MPJPE (protocol #1 metric)
        p2_error = np.zeros((sample_num, joint_num, 3)) # MPJPE (protocol #2 metroc)
        p1_error_action = [ [] for _ in range(len(self.action_idx)) ] # PA MPJPE for each action
        p2_error_action = [ [] for _ in range(len(self.action_idx)) ] # MPJPE error for each action
        pred_to_save = []
        for n in range(sample_num):
            
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_center = gt['center_cam']
            gt_3d_kpt = gt['joint_cam']
            gt_vis = gt['joint_vis'].copy()

            # restore coordinates to original space
            pre_2d_kpt = preds[n].copy()
            pre_2d_kpt[:,0], pre_2d_kpt[:,1], pre_2d_kpt[:,2] = warp_coord_to_original(pre_2d_kpt, bbox, gt_3d_center)

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1,500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3,joint_num))
                tmpkps[0,:], tmpkps[1,:] = pre_2d_kpt[:,0], pre_2d_kpt[:,1]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, filename + '_output.jpg'), tmpimg)

            # back project to camera coordinate system
            pre_3d_kpt = np.zeros((joint_num,3))
            pre_3d_kpt[:,0], pre_3d_kpt[:,1], pre_3d_kpt[:,2] = pixel2cam(pre_2d_kpt, f, c)

            vis = False
            if vis:
                vis_3d_skeleton(pre_3d_kpt, gt_vis, self.skeleton, filename)

            # root joint alignment
            pre_3d_kpt = pre_3d_kpt - pre_3d_kpt[self.root_idx]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]

            # rigid alignment for PA MPJPE (protocol #1)
            pre_3d_kpt_align = rigid_align(pre_3d_kpt, gt_3d_kpt)
 
            # prediction save
            pred_to_save.append({'pred': pre_3d_kpt,
                                 'align_pred': pre_3d_kpt_align,
                                 'gt': gt_3d_kpt})
           
            # error save
            p1_error[n] = np.power(pre_3d_kpt_align - gt_3d_kpt,2) # PA MPJPE (protocol #1)
            p2_error[n] = np.power(pre_3d_kpt - gt_3d_kpt,2)  # MPJPE (protocol #2)

            img_name = gt['img_path']
            action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
            p1_error_action[action_idx].append(p1_error[n].copy())
            p2_error_action[action_idx].append(p2_error[n].copy())


        # total error calculate
        p1_error = np.take(p1_error, self.eval_joint, axis=1)
        p2_error = np.take(p2_error, self.eval_joint, axis=1)
        p1_error = np.mean(np.power(np.sum(p1_error,axis=2),0.5))
        p2_error = np.mean(np.power(np.sum(p2_error,axis=2),0.5))

        p1_eval_summary = 'Protocol #1 error (PA MPJPE) >> %.2f' % (p1_error)
        p2_eval_summary = 'Protocol #2 error (MPJPE) >> %.2f' % (p2_error)
        print()
        print(p1_eval_summary)
        print(p2_eval_summary)

        # error for each action calculate
        p1_action_eval_summary = 'Protocol #1 error (PA MPJPE) for each action: \n'
        for i in range(len(p1_error_action)):
            err = np.array(p1_error_action[i])
            err = np.take(err, self.eval_joint, axis=1)
            err = np.mean(np.power(np.sum(err,axis=2),0.5))

            action_name = self.action_name[i]
            p1_action_eval_summary += (action_name + ': %.2f\n' % err)

            
        p2_action_eval_summary = 'Protocol #2 error (MPJPE) for each action: \n'
        for i in range(len(p2_error_action)):
            err = np.array(p2_error_action[i])
            err = np.take(err, self.eval_joint, axis=1)
            err = np.mean(np.power(np.sum(err,axis=2),0.5))

            action_name = self.action_name[i]
            p2_action_eval_summary += (action_name + ': %.2f\n' % err)
        print()
        print(p1_action_eval_summary)
        print(p2_action_eval_summary)
       
        # result save
        f_pred_3d_kpt = open(osp.join(result_dir, 'pred_3d_kpt.txt'), 'w')
        f_pred_3d_kpt_align = open(osp.join(result_dir, 'pred_3d_kpt_align.txt'), 'w')
        f_gt_3d_kpt = open(osp.join(result_dir, 'gt_3d_kpt.txt'), 'w')
        for i in range(len(pred_to_save)):
            for j in range(joint_num):
                for k in range(3):
                    f_pred_3d_kpt.write('%.3f ' % pred_to_save[i]['pred'][j][k])
                    f_pred_3d_kpt_align.write('%.3f ' % pred_to_save[i]['align_pred'][j][k])
                    f_gt_3d_kpt.write('%.3f ' % pred_to_save[i]['gt'][j][k])
            f_pred_3d_kpt.write('\n')
            f_pred_3d_kpt_align.write('\n')
            f_gt_3d_kpt.write('\n')
        f_pred_3d_kpt.close()
        f_pred_3d_kpt_align.close()
        f_gt_3d_kpt.close()

        f_eval_result = open(osp.join(result_dir, 'eval_result.txt'), 'w')
        f_eval_result.write(p1_eval_summary)
        f_eval_result.write('\n')
        f_eval_result.write(p2_eval_summary)
        f_eval_result.write('\n')
        f_eval_result.write(p1_action_eval_summary)
        f_eval_result.write('\n')
        f_eval_result.write(p2_action_eval_summary)
        f_eval_result.write('\n')
        f_eval_result.close()




