# -*- coding: utf-8 -*-
"""
Note: description below is copied from Jupyter NB description; UPDATE NEEDED.

Prototype tracker, designed for tracking people in retail store video data.

This prototype uses 3 base algorithms. They have been modified to work together and to create a full person tracking algorithm:
1. YOLOv3, for person detection. See https://github.com/AlexeyAB/darknet and https://dev.to/kojikanao/yolo-on-google-colab-4b8e.
2. Deepsort, for initial person tracking. See https://github.com/abhyantrika/nanonets_object_tracking.
3. Reidentification algorithm, for improved person tracking, especially between cameras and without ID swaps. See https://github.com/layumi/Person_reID_baseline_pytorch.git.

To use:
1. Specify input_path in the cell below. This should be the directory where the video for tracking is located. It can be formatted as .avi or .mp4, or given as individual .jpg frames.
2. Specify framerate.
3. Comment the two Google Drive mounting lines in the cell below if not using that.
4. The third cell is optional, for emptying gallery/query folders (i.e. discarding the history of people the tracker has seen and identified).
5. Run the rest.

Besides the gallery folder, this will generate an output video in the location the input directory is stored, \<input directory name\>_tracked.avi
"""
from __future__ import print_function, division

import argparse

import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--video_id')
    args = parser.parse_args()
    
    # Specify path
    input_path = args.path
    video_id = int(args.video_id)
    
    # Specify framerate.
    frame_rate = 15.0
    
    import os
    base_directory = os.path.split(input_path)[0] # for use constructing some paths
    
      

    
    import glob
    import cv2
    import re
    
    # functions for sorting paths in natural way (using number in the filename)
    def int_detector(text):
        return int(text) if text.isdigit() else text
    def natural_sorter(text):
        return [ int_detector(c) for c in re.split('(\d+)',text) ]
    
    
    video_path = glob.glob(input_path + '/*.avi')
    video_path.extend(glob.glob(input_path + '/*.mp4'))
    

      

    
    
    
    
    # This cell sets up reID paramters and functions.
    

    
    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.autograd import Variable
    import torch.backends.cudnn as cudnn
    import numpy as np
    import torchvision
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, models, transforms
    import time
    import os
    import scipy.io
    import yaml
    import math
    import numpy as np
    from collections import Counter
    from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
    
    
    #fp16
    try:
        from apex.fp16_utils import *
    except ImportError: # will be 3.x series
        print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    ######################################################################
    # Options
    # --------
    
    
    class Args:
      fp16 = ''
      PCB = ''
      use_dense = ''
      use_NAS = ''
      stride = ''
     
    
    opt = Args()
    
    opt.multi = False
    opt.batchsize = 128
    
    config_path = os.path.join('./model/PCB','opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream)
    opt.fp16 = config['fp16'] 
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_NAS = False
    opt.stride = config['stride']
    
    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else: 
        opt.nclasses = 751 
    
    str_ids = '0'
    gpu_ids = [0]
    #which_epoch = opt.which_epoch
    name = 'PCB'
    test_dir = base_directory
    
    
    
    print('We use the scale: %s'%'1')
    str_ms = '1'
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))
    
    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    data_transforms = transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop        
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.ToTensor()(crop) 
              #      for crop in crops]
               # )),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
              #       for crop in crops]
              # ))
    ])
    
    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    
    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        #save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        save_path = os.path.join('./model',name,'net_last.pth')
        network.load_state_dict(torch.load(save_path))
        return network
    
    
    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip
    
    def extract_feature(model,dataloaders):
        features = torch.FloatTensor()
        count = 0
        for data in dataloaders:
            img, label = data
            n, c, h, w = img.size()
            count += n
            print(count)
            ff = torch.FloatTensor(n,512).zero_().cuda()
            if opt.PCB:
                ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
    
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
    
            features = torch.cat((features,ff.data.cpu()), 0)
        return features
    
    def extract_feature_query(model,dataloaders):
        features = torch.FloatTensor()
        count = 0
        for data in dataloaders:
            img = data
            n, c, h, w = img.size()
            count += n
            print(count)
            ff = torch.FloatTensor(n,512).zero_().cuda()
            if opt.PCB:
                ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
    
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
    
            features = torch.cat((features,ff.data.cpu()), 0)
        return features
    
    def get_id(img_path):
      camera_id = []
      labels = []
      for path, v in img_path:
          filename = path.split('/')[-2:]
          label = filename[0][6:]
          camera = [0]
          if label[0:2]=='-1':
              labels.append(-1)
          else:
              labels.append(int(label))
          camera_id.append(int(camera[0]))
      return camera_id, labels
    
    
    # Evaluation function. Compares query features (qf) with gallery features (gf) and looks for a match.
    def evaluate2(qf,gf,gl):
        query = qf.view(-1,1)
        score = torch.mm(gf,query) # cosine similarity (note features are already normalized)
        score = score.squeeze(1).cpu()
        score = score.numpy()
    
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1][:5] # large to small, top 5
        gl_indexed = [gl[idx] for idx in index]
        c = Counter(gl_indexed)
        value, count = c.most_common()[0] # take majority vote of the top 5
        indexer2 = [index2 for index2 in index if gl[index2] == value]
        overall = sum(score[indexer2]) / count
        print(overall)
        if overall > .7: # if confidence is reasonably high, return reID prediction
          return value
        else: # otherwise, return signal to use tracker prediction
          return -1
    
    # Dataset class to load region of interest (Roi) from tracker into the reID algorithm
    class RoiDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform
            
        def __getitem__(self, index):
            x = self.data[index]
            
            if self.transform:
                x = self.transform(x)
            
            return x
        
        def __len__(self):
            return len(self.data)
    
    # Transforms for use with the above class.
    data_transforms_query = transforms.Compose([
            transforms.ToPILImage(),                                    
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if opt.PCB:
        data_transforms_query = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        
        
        
        
        
        
    # This cell sets up parameters and functions for the tracker.
    # This is copied from deepsort with the only modification being set max_age to 5 on the tracker (from 40)
    
    from deep_sort.deep_sort import nn_matching
    from deep_sort.deep_sort.tracker import Tracker 
    from deep_sort.application_util import preprocessing as prep
    from deep_sort.application_util import visualization
    from deep_sort.deep_sort.detection import Detection
    
    import numpy as np
    
    import matplotlib.pyplot as plt
    
    import torch
    import torchvision
    from scipy.stats import multivariate_normal
    
    def get_gaussian_mask():
    	#128 is image size
    	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    	xy = np.column_stack([x.flat, y.flat])
    	mu = np.array([0.5,0.5])
    	sigma = np.array([0.22,0.22])
    	covariance = np.diag(sigma**2) 
    	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
    	z = z.reshape(x.shape) 
    
    	z = z / z.max()
    	z  = z.astype(np.float32)
    
    	mask = torch.from_numpy(z)
    
    	return mask
    
    
    
    
    class deepsort_rbc():
    	def __init__(self,wt_path=None):
    		#loading this encoder is slow, should be done only once.
    		#self.encoder = generate_detections.create_box_encoder("deep_sort/resources/networks/mars-small128.ckpt-68577")		
    		if wt_path is not None:
    			self.encoder = torch.load(wt_path)			
    		else:
    			self.encoder = torch.load('ckpts/model640.pt')
    			
    		self.encoder = self.encoder.cuda()
    		self.encoder = self.encoder.eval()
    		print("Deep sort model loaded")
    
    		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",.5 , 100)
    		self.tracker = Tracker(self.metric, max_age = 5) 
    
    		self.gaussian_mask = get_gaussian_mask().cuda()
    
    
    		self.transforms = torchvision.transforms.Compose([ \
    				torchvision.transforms.ToPILImage(),\
    				torchvision.transforms.Resize((128,128)),\
    				torchvision.transforms.ToTensor()])
    
    
    
    	def reset_tracker(self):
    		self.tracker= Tracker(self.metric)
    
    	#Deep sort needs the format `top_left_x, top_left_y, width,height
    	
    	def format_yolo_output( self,out_boxes):
    		for b in range(len(out_boxes)):
    			out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
    			out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
    		return out_boxes				
    
    	def pre_process(self,frame,detections):	
    
    		transforms = torchvision.transforms.Compose([ \
    			torchvision.transforms.ToPILImage(),\
    			torchvision.transforms.Resize((128,128)),\
    			torchvision.transforms.ToTensor()])
    
    		crops = []
    		for d in detections:
    
    			for i in range(len(d)):
    				if d[i] <0:
    					d[i] = 0	
    
    			img_h,img_w,img_ch = frame.shape
    
    			xmin,ymin,w,h = d
    
    			if xmin > img_w:
    				xmin = img_w
    
    			if ymin > img_h:
    				ymin = img_h
    
    			xmax = xmin + w
    			ymax = ymin + h
    
    			ymin = abs(int(ymin))
    			ymax = abs(int(ymax))
    			xmin = abs(int(xmin))
    			xmax = abs(int(xmax))
    
    			try:
    				crop = frame[ymin:ymax,xmin:xmax,:]
    				crop = transforms(crop)
    				crops.append(crop)
    			except:
    				continue
    
    		crops = torch.stack(crops)
    
    		return crops
    
    	def extract_features_only(self,frame,coords):
    
    		for i in range(len(coords)):
    			if coords[i] <0:
    				coords[i] = 0	
    
    
    		img_h,img_w,img_ch = frame.shape
    				
    		xmin,ymin,w,h = coords
    
    		if xmin > img_w:
    			xmin = img_w
    
    		if ymin > img_h:
    			ymin = img_h
    
    		xmax = xmin + w
    		ymax = ymin + h
    
    		ymin = abs(int(ymin))
    		ymax = abs(int(ymax))
    		xmin = abs(int(xmin))
    		xmax = abs(int(xmax))
    		
    		crop = frame[ymin:ymax,xmin:xmax,:]
    		#crop = crop.astype(np.uint8)
    
    		#print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)
    
    		crop = self.transforms(crop)
    		crop = crop.cuda()
    
    		gaussian_mask = self.gaussian_mask
    
    		input_ = crop * gaussian_mask
    		input_ = torch.unsqueeze(input_,0)
    
    		features = self.encoder.forward_once(input_)
    		features = features.detach().cpu().numpy()
    
    		corrected_crop = [xmin,ymin,xmax,ymax]
    
    		return features,corrected_crop
    
    
    	def run_deep_sort(self, frame, out_scores, out_boxes):
    
    		if out_boxes==[]:			
    			self.tracker.predict()
    			print('No detections')
    			trackers = self.tracker.tracks
    			return trackers
    
    		detections = np.array(out_boxes)
    		#features = self.encoder(frame, detections.copy())
    
    		processed_crops = self.pre_process(frame,detections).cuda()
    		processed_crops = self.gaussian_mask * processed_crops
    
    		features = self.encoder.forward_once(processed_crops)
    		features = features.detach().cpu().numpy()
    
    		if len(features.shape)==1:
    			features = np.expand_dims(features,0)
    
    
    		dets = [Detection(bbox, score, feature) \
    					for bbox,score, feature in\
    				zip(detections,out_scores, features)]
    
    		outboxes = np.array([d.tlwh for d in dets])
    
    		outscores = np.array([d.confidence for d in dets])
    		indices = prep.non_max_suppression(outboxes, 0.8,outscores)
    		
    		dets = [dets[i] for i in indices]
    
    		self.tracker.predict()
    		self.tracker.update(dets)	
    
    		return self.tracker,dets
        
        
        
        
    # Now we run the tracker + reID
    
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import torchvision.utils
    import random
    from PIL import Image
    import PIL.ImageOps    
    from torch.autograd import Variable
    from torch import optim
    import torch.nn.functional as F
    import cv2,pickle,sys
    import shutil
    
    #from deepsort import *
    
    
    def get_gt(image,frame_id,gt_dict):
    
    	if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
    		return None,None
    		#return None,None,None
    
    	frame_info = gt_dict[frame_id]
    
    	detections = []
    	ids = []
    	out_scores = []
    	for i in range(len(frame_info)):
    
    		coords = frame_info[i]['coords']
    
    		x1,y1,w,h = coords
    		x2 = x1 + w
    		y2 = y1 + h
    
    		xmin = min(x1,x2)
    		xmax = max(x1,x2)
    		ymin = min(y1,y2)
    		ymax = max(y1,y2)	
    
    		detections.append([x1,y1,w,h])
    		out_scores.append(frame_info[i]['conf'])
    
    	return detections,out_scores
    
    
    def get_dict(filename):
    	with open(filename) as f:	
    		d = f.readlines()
    
    	d = list(map(lambda x:x.strip(),d))
    
    	last_frame = int(d[-1].split(',')[0])
    
    	gt_dict = {x:[] for x in range(last_frame+1)}
    
    	for i in range(len(d)):
    		a = list(d[i].split(','))
    		a = list(map(float,a))	
    
    		coords = a[2:6]
    		confidence = a[6]
    		gt_dict[a[0]].append({'coords':coords,'conf':confidence})
    
    	return gt_dict
    
    def get_mask(filename):
    	mask = cv2.imread(filename,0)
    	mask = mask / 255.0
    	return mask
    
    
    
    
    
    #Load detections for the video. Options available: yolo,ssd and mask-rcnn
    filename = 'det/det_yolo3.txt'
    gt_dict = get_dict(filename)
    
    video_path
    cap = cv2.VideoCapture(video_path[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #Initialize deep sort.
    deepsort = deepsort_rbc()
    
    frame_id = 1
    
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    clip_width = 192
    clip_height = 384
    
    
    #initialize reID
    base_directory = os.path.split(input_path)[0]
    data_dir = test_dir
    use_gpu = torch.cuda.is_available()
    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride)
    
    if opt.PCB:
        model_structure = PCB(opt.nclasses)
    
    #if opt.fp16:
    #    model_structure = network_to_half(model_structure)
    
    model = load_network(model_structure)
    
    # Remove the final fc layer and classifier layer
    if opt.PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()
    
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    
    
    gallery_loaded = False
    
    if video_id == 0:
        det_counter = 0
    else:
        det_counter = 100
    ######################################################################
    
    id_set = set()
    
    output_list = []
    
    # Loop over frames and apply tracking and reID
    while True:
      print(frame_id)		
      #if frame_id > 300:
      #  break
      ret,frame = cap.read()
      if ret is False:
        frame_id+=1
        break	
    
      frame = frame.astype(np.uint8)
    
      detections,out_scores = get_gt(frame,frame_id,gt_dict)
    
      if detections is None:
        print("No dets")    
        frame_id+=1
        continue
      det_counter += 1
    
      detections = np.array(detections)
      out_scores = np.array(out_scores) 
    
      tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)
    
      track_reID = []
      bbox_list = []
    
      if det_counter < 10:
        for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
            continue
          
          bbox = track.to_tlbr() #Get the corrected/predicted bounding box
          roi = frame[np.maximum(int(bbox[1]),0):np.maximum(int(bbox[3]),0), np.maximum(int(bbox[0]),0):np.maximum(int(bbox[2]),0)]
          roi = cv2.resize(roi,(clip_width,clip_height),interpolation=cv2.INTER_CUBIC)
          if not os.path.isdir(base_directory + '/gallery/person{}'.format(track.track_id)):
            os.mkdir(base_directory + '/gallery/person{}'.format(track.track_id))
            id_set.add(track.track_id)
          cv2.imwrite( base_directory + '/gallery/person{}/{}.jpg'.format(track.track_id,frame_id), roi )
          output_list.append('{} {} {}'.format(frame_id,track.track_id,bbox))
    
    
      else:
    
        if not gallery_loaded:
          image_datasets = {'gallery': datasets.ImageFolder( os.path.join(data_dir,'gallery') ,data_transforms)}
          dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery']}
    
          gallery_path = image_datasets['gallery'].imgs
          gallery_cam,gallery_label = get_id(gallery_path)
          
          with torch.no_grad():
            gallery_feature = extract_feature(model,dataloaders['gallery'])
    
          gallery_loaded = True
    
        roi_list = []
        for track_idx, track in enumerate(tracker.tracks):
          if not track.is_confirmed() or track.time_since_update > 1:
            continue
          
          bbox = track.to_tlbr() #Get the corrected/predicted bounding box
          bbox_list.append(bbox)
          roi_temp = frame[np.maximum(int(bbox[1]),0):np.maximum(int(bbox[3]),0), np.maximum(int(bbox[0]),0):np.maximum(int(bbox[2]),0)]
          roi_temp = cv2.resize(roi_temp,(clip_width,clip_height),interpolation=cv2.INTER_CUBIC)
          roi_list.append(roi_temp)
        
        roi_converted = []
        for roi_img in roi_list:
          roi_converted.append(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        if roi_converted:
          roi = np.stack(roi_converted,axis=0)
        else:
          frame_id += 1
          continue
    
        query_dataset = RoiDataset(roi, data_transforms_query)
        query_dataloader = torch.utils.data.DataLoader(query_dataset, batch_size=opt.batchsize,
                                                shuffle=False, num_workers=16)
    
        # Extract feature
        with torch.no_grad():
            query_feature = extract_feature_query(model,query_dataloader)
    
        proposed_labels = []
        for i in range(query_feature.shape[0]):
          proposed_labels.append(evaluate2(query_feature[i],gallery_feature,gallery_label))
        print("Proposed labels:", proposed_labels)
    
        roi_counter = 0
        query_labels = []
        for tr_idx, pl in enumerate(proposed_labels):
          if pl != -1: # reID is confident. We take its label as our estimate.
            cv2.imwrite( base_directory + "/gallery/person{}/{}.jpg".format(pl,frame_id), roi_list[roi_counter] )
            track_reID.append(pl)
            query_labels.append(pl)
            #tracker.tracks[track_idx].track_id = pl
            output_list.append('{} {} {}'.format(frame_id,pl,bbox_list[roi_counter]))
          else: # reID is NOT confident. We take tracker label.
            new_id = tracker.tracks[tr_idx].track_id + video_id
            #new_id = max(id_set) + 1
            #id_set.add(new_id)
            if not os.path.isdir(base_directory + "/gallery/person{}".format(new_id)):
              os.mkdir(base_directory + "/gallery/person{}".format(new_id))
            cv2.imwrite( base_directory + "/gallery/person{}/{}.jpg".format(new_id,frame_id), roi_list[roi_counter] )
            track_reID.append(new_id)
            query_labels.append(new_id)
            output_list.append('{} {} {}'.format(frame_id,new_id,bbox_list[roi_counter]))
          
          roi_counter += 1
    
        gallery_feature = torch.cat((gallery_feature, query_feature), 0)
        gallery_label = gallery_label + query_labels 
    
    
      frame_id+=1
    
    with open(input_path + '/output2.txt', 'w') as filehandle: 
      filehandle.writelines("%s\n" % output_line for output_line in output_list)
