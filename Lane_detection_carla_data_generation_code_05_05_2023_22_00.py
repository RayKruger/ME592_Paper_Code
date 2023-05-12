# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:53:09 2023

@author: RAY
"""
#Note Run scrypt in administrative mode
# =============================================================================
# Resources
# =============================================================================

# https://carla.readthedocs.io/en/latest/tuto_first_steps/
# https://carla.readthedocs.io/en/latest/tutohttp://localhost:8889/notebooks/Carla_Image_capture.ipynb#_G_bounding_boxes/
# https://carla.readthedocs.io/en/0.9.9/tuto_G_retrieve_data/#weather-setting
# https://arijitray1993.github.io/CARLA_tutorial/
# =============================================================================
# 
# =============================================================================
import glob
import os
import sys
import time
import json
import io
import shutil
import os
import socket
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# =============================================================================
# import packages
# =============================================================================
import carla

import argparse
import logging
import random

import subprocess
import time as tm
import pygame
from pygame.locals import *
import numpy as np
import cv2
import copy

import queue
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import struct
import signal

import pycocotools
from pycocotools import mask 
import skimage.measure as measure
from skimage.measure import approximate_polygon
from numba import jit
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats     

import random
#%%  
# =============================================================================
# clear folders and delete json
# =============================================================================
Dir='C:/Users/RAY/Desktop/Lane_Detection_Project/Example_code/tutorial/output/'
folders = ['DM', 'RGBLF', 'RGBRF', 'SEG','Mask']
for folder in folders:
    folder_path = os.path.join(Dir, folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')    
# delete the json   
try:         
    os.remove(Dir+'lane_coco_annotations.json')         
except:
    pass
tm.sleep(5)        
print('done deteting files')

# reset network socket port that is used by carla
port = 2000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('localhost', port))
s.close()


# =============================================================================
# Lounch Carla 
# =============================================================================
Dir='C:/Users/RAY/Desktop/Lane_Detection_Project/Example_code/tutorial/'
# start carla
if 1==1:      
    carla_executable = 'C:/Users/RAY/Desktop/Lane_Detection_Project/CARLA_0.9.14/WindowsNoEditor/CarlaUE4.exe'
    carla_args = ['-windowed', '-ResX=800', '-ResY=600', '-carla-server', '-benchmark', '-fps=60']
    carla_process = subprocess.Popen([carla_executable] + carla_args)
    carla_pid = carla_process.pid
    print('Starting...Carla server')
    tm.sleep(15)
    print('Sucsesfully started Carla server')

#%%    
# =============================================================================
# Helper functions
# =============================================================================
# --------------
# Carla standard mask segmentations 
# --------------
object_list = dict()
object_list['RoadLine'] = np.uint8([[[157, 234,  50]]])
object_list['Road'] = np.uint8([[[128, 64,  128]]])
object_list['Sidewalk'] = np.uint8([[[244, 35,   232]]])
object_list['car'] = np.uint8([[[ 0, 0, 142]]])
object_list['building'] = np.uint8([[[70, 70, 70]]])        
object_list['pedestrian'] = np.uint8([[[220, 20, 60]]]) 

object_list['building'] = np.uint8([[[70, 70, 70]]])
object_list['vegetation'] = np.uint8([[[107, 142, 35]]])
object_list['fence'] = np.uint8([[[ 190, 153, 153]]])
object_list['traffic_sign'] = np.uint8([[[220, 220, 0]]])
object_list['pole'] = np.uint8([[[153, 153, 153]]])
object_list['wall'] = np.uint8([[[102, 102, 156]]])
# more at https://carla.readthedocs.io/en/0.9.9/ref_sensors/        
        


# --------------
# reate an empty dictionary to store the COCO format data before the game loop
# --------------                

coco_data = {
     "info": {
         "description": "ME592 Carla Lanes dataset",
         "version": "1.0",
         "year": 2023,
         "contributor": "Ray Kruger",
         "date_created": "2023/04/27"
     },
     
    "licenses": [
         {
             "id": 1,
             "name": "CC BY 4.0",
             "url": "https://creativecommons.org/licenses/by/4.0/"
         }
     ],
    
     "images": [],
     
     "annotations": [],
     
     "categories": [
         {
         "id": 1,
         "name": "lanes",
         "supercategory": "none"
         }
     ] # add more if needed
 }
# =============================================================================
# 
# =============================================================================
#Segmentation colour palet C++ code on Carla github https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
CITYSCAPES_PALETTE_MAP = np.array([
    [0, 0, 0], # unlabeled
    [128, 64, 128], # road 
    [244, 35, 232], # sidewalk 
    [70, 70, 70], # building 
    [102, 102, 156], # wall 
    [190, 153, 153], # fence 
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [110, 190, 160],
    [170, 120, 50],
    [55, 90, 80],
    [45, 60, 150],
    [157, 234, 50],
    [81, 0, 81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180]], dtype=np.uint8)

def apply_cityscapes_palette(image_data, height, width):       
    image_data = np.reshape(image_data, (height, width, 4))
    indices = image_data[:, :, 2]   
    img_semseg_bgra = np.zeros((height, width, 4), dtype=np.uint8)
    img_semseg_bgra[:, :, :3] = CITYSCAPES_PALETTE_MAP[indices] # directly map the indices to their corresponding colors    
    img_semseg_bgra[:, :, 3] = 255 # Add alpha channel value
    return img_semseg_bgra
JPEG_QUALITY=80
# =============================================================================
# 
# =============================================================================
def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        world = client.get_world()
        ego_vehicle = None
        ego_cam = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_gnss = None
        ego_imu = None
        
        # --------------
        # set town, wheather and spown point
        # --------------
                
        #weather = carla.WeatherParameters(
        #    cloudiness=10.0,
        #    precipitation=10.0,
        #    sun_altitude_angle=30.0
        #)
        #world.set_weather(weather)
        def set_town_by_index(index,world,client):
            # Define a list of available towns
            towns = [
                'Town01', #0  # A small, simple town with a river and several bridges.
                'Town02', #1  # A small simple town with a mixture of residential and commercial buildings.
                'Town03', #2 # A larger, urban map with a roundabout and large junctions.
                'Town04', #3 # A small town embedded in the mountains with a special "figure of 8" infinite highway.
                'Town05', #4 # Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
                'Town06', #5 # Long many lane highways with many highway entrances and exits. It also has a Michigan left.
                #'Town07', # # A rural environment with narrow roads, corn, barns and hardly any traffic lights.
                #'Town08',  # Secret "unseen" town used for the Leaderboard challenge
                #'Town09',  # Secret "unseen" town used for the Leaderboard challenge
                'Town10', #6  # A downtown urban environment with skyscrapers, residential buildings and an ocean promenade.
                #'Town11', #8  # A Large Map that is undecorated. Serves as a proof of concept for the Large Maps feature.
                'Town12', #7  # A Large Map with numerous different regions, including high-rise, residential and rural environments.
            ]
            world=client.load_world(towns[index])

            
        
        def set_weather_by_index(index,world):                
            # Define a list of available weather conditions        
            weather_conditions = [
            carla.WeatherParameters.ClearNoon, # 0
            carla.WeatherParameters.CloudyNoon, # 1
            carla.WeatherParameters.WetNoon, # 2
            carla.WeatherParameters.WetCloudyNoon, # 3
            carla.WeatherParameters.MidRainyNoon, # 4
            carla.WeatherParameters.HardRainNoon, # 5
            carla.WeatherParameters.SoftRainNoon, # 6
            carla.WeatherParameters.ClearSunset, # 7
            carla.WeatherParameters.CloudySunset, # 8
            carla.WeatherParameters.WetSunset, # 9
            carla.WeatherParameters.WetCloudySunset, # 10
            carla.WeatherParameters.MidRainSunset, # 11
            carla.WeatherParameters.HardRainSunset, # 12
            carla.WeatherParameters.SoftRainSunset, # 13
            ]
            
            # Set the world weather to the weather condition at the specified index
            world.set_weather(weather_conditions[index])
            
        #randomly smaple from wheather a specified omout of time
        def generate_wheather_samples(num_samples):
            percentages=np.zeros([14])
            percentages[0]=40 # ClearNoon
            percentages[2]=10 # CloudyNoon
            percentages[5]=5 # HardRainNoon
            
            percentages[7]=20 # ClearSunset           
            percentages[8]=10 # CloudySunset
            percentages[12]=10 # HardRainSunset
            percentages[13]=5 # SoftRainSunset
                        
            idx = np.arange(num_samples)
        
            # generate a list of samples for each index based on the percentages
            samples = []
            for i in range(len(percentages)):
                num_samples_i = int(percentages[i] / 100 * num_samples)
                samples += [idx[i]] * num_samples_i
        
            # fill in any remaining samples with a randomly selected index
            while len(samples) < num_samples:
                samples.append(random.randint(0, num_samples - 1))
                
            random.shuffle(samples)
            return samples
            
        #set_town_by_index(4, world,client)  #MAX_IDX=7
        weather_idx=0  
        town_idx=0
        #set_weather_by_index(12, world) #MAX_IDX=12   
        #world.set_weather(carla.WeatherParameters.HardRainSunset)
        def spown_new_world(world,client):
            # --------------
            # set data recording mode and traffic manager
            # --------------        
            
            if 1==1: #syncronous mode
                traffic_manager = client.get_trafficmanager(8000)
                traffic_manager.set_synchronous_mode(True)      
                settings = world.get_settings()
                settings.fixed_delta_seconds = 1 #0.05  # Set the fixed time step for the simulation, for example, 0.05 seconds (20 FPS)
                settings.synchronous_mode = True
                world.apply_settings(settings)
    
                  
            # --------------
            # Start recording
            # --------------
            
            client.start_recorder('~/tutorial/recorder/recording01.log')
    
            # --------------
            # Spawn ego vehicle
            # --------------
            
            ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name','ego')
            print('\nEgo role_name is set')
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_color = ego_bp.get_attribute('color').recommended_values[0]
            ego_bp.set_attribute('color',ego_color)
            print('\nEgo color is set')
    
            def set_spawn_point(world,Random=False):
                spawn_points = world.get_map().get_spawn_points()
                number_of_spawn_points = len(spawn_points)
        
                if 0 < number_of_spawn_points:
                    if Random==True:
                        #random.shuffle(spawn_points)
                        idx_spawn=np.random.randint(len(spawn_points))
                        ego_transform = spawn_points[idx_spawn]                    
                    else:    
                        ego_transform = spawn_points[2]
                    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
                    ego_vehicle.set_autopilot(True)  
                    print('\nEgo is spawned')
                else: 
                    logging.warning('Could not found any spawn points')
                    
                return  [ego_vehicle,ego_transform] 
    
            [ego_vehicle,ego_transform]=set_spawn_point(world,Random=True) 
    
            
            # Spawn 50 vehicles randomly distributed throughout the map        
            vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*') # Get the blueprint library and filter for the vehicle blueprints
            spawn_points = world.get_map().get_spawn_points()
            for i in range(0,50):
                world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
            for vehicle in world.get_actors().filter('*vehicle*'):
                vehicle.set_autopilot(True)    
                traffic_manager.update_vehicle_lights(vehicle, True)
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(100) #All vehicles inside this radius will have physics enabled; vehicles outside of the radius will have physics disabled.        
            traffic_manager.set_respawn_dormant_vehicles(True) 
            traffic_manager.set_boundaries_respawn_dormant_vehicles(50,700)
                
            # --------------
            # save_rgb_image
            # --------------
            def save_rgb_image(image,file_name):
                #The images are sent by the server as stream of  32 bit BGRA array of  bytes. It's up to the users to parse the images and convert them to the desired format. 
                image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                image_data = np.reshape(image_data, (image.height,image.width, 4)) # rows, colums                              
                image_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB) # Extract the RGB channels and convert the image to the correct color space
                cv2.imwrite(file_name, image_rgb,[int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                
                #image_rgb = image_data[:, :, :3] #Drop A from BGRA
                #image_rgb = image_rgb[:, :, ::-1] #BGR->RGB
                #image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB) # shape (height, width, 4) 
                
            # --------------
            # save_depth_image
            # --------------   
            def save_depth_image(image,file_name):
                #The image codifies the depth in 3 channels of the RGB color space, from less to more significant bytes: R -> G -> B. The actual distance in meters can be decoded with
                image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                image_data = np.reshape(image_data, (image.height,image.width, 4)) # rows, colums                 
                image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB) # shape (height, width, 4)
                R=image_rgb[:,:,0].astype(np.uint32)
                G=image_rgb[:,:,1].astype(np.uint32)
                B=image_rgb[:,:,2].astype(np.uint32)
                normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                Z_meters = 1000 * normalized
              
                # Map depths in meters to 0-256 BW range
                Z_near=1; Z_far=80 
                Z_range = Z_far - Z_near
                depth_mapped = ((Z_meters- Z_near ) / Z_range) * 255  
                depth_mapped = np.clip(depth_mapped, 0, 255).astype(np.uint8)           
                cv2.imwrite(file_name, depth_mapped,[int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                
    
            # --------------
            # LF RGB camera sensor to ego vehicle. 
            # --------------       
            LF_bp = None
            LF_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            LF_bp.set_attribute("image_size_x",str(640))#960
            LF_bp.set_attribute("image_size_y",str(480))#540
            LF_bp.set_attribute("fov",str(105))
            #LF_bp.set_attribute('sensor_tick', '0.1')
            #LF_bp.set_attribute('enable_postprocess_effects', 'False')
            LF_location = carla.Location(1.3,-0.1,1.3)
            LF_rotation = carla.Rotation(0,0,0)
            LF_transform = carla.Transform(LF_location,LF_rotation)
            LF_cam = world.spawn_actor(LF_bp,LF_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)            
            LF_cam.listen(lambda image:save_rgb_image(image, Dir+"/output/RGBLF/LF_%06d.jpg" %(image.frame)))
            
            # --------------
            # RF RGB camera sensor to ego vehicle. 
            # --------------
            
            RF_bp = None
            RF_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            RF_bp.set_attribute("image_size_x",str(640))#960
            RF_bp.set_attribute("image_size_y",str(480))#540
            RF_bp.set_attribute("fov",str(105))
            #RF_bp.set_attribute('sensor_tick', '0.1')
            #RF_bp.set_attribute('enable_postprocess_effects', 'False')
            RF_location = carla.Location(1.32,0.1,1.3)
            RF_rotation = carla.Rotation(0,0,0)
            RF_transform = carla.Transform(RF_location,RF_rotation)
            RF_cam = world.spawn_actor(RF_bp,RF_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)            
            RF_cam.listen(lambda image:save_rgb_image(image, Dir+"/output/RGBRF/RF_%06d.jpg" %(image.frame)))
            
            # --------------
            # Add a Depth camera to ego vehicle. 
            # --------------
            depth_bp  = None
            depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
            depth_bp.set_attribute("image_size_x",str(640))#960
            depth_bp.set_attribute("image_size_y",str(480))#540
            depth_bp.set_attribute("fov",str(105))
            depth_location = carla.Location(1.32,-0.1,1.3)
            depth_rotation = carla.Rotation(0,0,0)
            depth_transform = carla.Transform(depth_location,depth_rotation)
            depth_cam = world.spawn_actor(depth_bp,depth_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            #depth_cam.listen(lambda image: image.save_to_disk(Dir+'/output/DM/DM_%.6d.jpg' % (image.frame),carla.ColorConverter.LogarithmicDepth))        
            depth_cam.listen(lambda image:save_depth_image(image, Dir+'/output/DM/DM_%.6d.png' % (image.frame)))     
                   
            # --------------
            # Get mask for class 
            # --------------
            def get_hsv_mask(seg_im, rgb_value):
                # rgb_value should be somethiing like np.uint8([[[70, 70, 70]]])                      
                hsv_value = cv2.cvtColor(rgb_value, cv2.COLOR_RGB2HSV) # seg_im should be in HSV already              
                tol=100
                hsv_low = np.array([[[hsv_value[0][0][0]-tol, hsv_value[0][0][1], hsv_value[0][0][2]-tol]]])
                hsv_high = np.array([[[hsv_value[0][0][0]+tol, hsv_value[0][0][1], hsv_value[0][0][2]+tol]]])    
                mask = cv2.inRange(seg_im, hsv_low, hsv_high)
                return mask
    
            def get_binary_greyscale_mask_from_hsv(mask):
                img_gray = mask#[:,:,2]    # Extract the Value (V) channel of the image
                thresh_val=10
                max_val=256
                ret, binary_mask = cv2.threshold(img_gray, thresh_val, max_val, cv2.THRESH_BINARY) 
                return binary_mask
            
            def get_binary_mask_from_greyscale(mask):
                return np.array(mask/255.0,dtype='int8')
                             
            # --------------
            # Get the bbdx for the class
            # --------------
            def get_bboxes_from_mask(mask):
                label_mask = measure.label(mask)
                props = measure.regionprops(label_mask)            
                return [prop.bbox for prop in props]
    
    
            # --------------
            # Create coco like json data anotations 
            # --------------
                    
            global lane_invasion_frame
            lane_invasion_frame=0
            def create_coco_annotation(image, bboxes, mask):
                #import pdb;pdb.set_trace()
                annotations = []
                for idx, bbox in enumerate(bboxes):
                    
                    # Create a binary mask for each object from global binary mask
                    object_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    object_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    
                    # Convert object binary mask to polygons
                    contours = measure.find_contours(object_mask, 0.5)
                    polygons = []
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        contour=np.round(contour,1)             
                        segmentation = contour.ravel().tolist()                   
                        polygons.append(segmentation)
                    
                        
                    # lane invasion code   =0              
                    global lane_invasion_frame                
                    if lane_invasion_frame==image.frame:
                        lane_invasion=1
                    else: 
                        lane_invasion=0
                
                    # Create object annotation
                    annotation = {
                        "id": len(annotations) + 1,
                        "image_id": image.frame,   #in image list it is called id
                        "category_id": 1,  # Assuming only one category for road lines
                        "segmentation": polygons,  # [x1, y1, x2, y2, ..., xn, yn]  # Polygon points for the mask
                        "area": bbox[2] * bbox[3],
                        "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]], #[x, y, width, height] format instead of the current format [y1, x1, y2, x2].
                        "iscrowd": 0,
                        "lane_invasion":lane_invasion
                    }
                    
                    # Add all segmented objects to object list
                    annotations.append(annotation)
                return annotations
            
            def filter_horizontal_movements(contour, threshold_angle=30):
                # If the angle is within threshold_angle degrees from the vertical line (measured from the y-axis), 
                new_contour = []        
                for i in range(1, len(contour)):
                    # Calculate the angle between the current point and the previous one
                    dx = contour[i][0][0] - contour[i-1][0][0]
                    dy = contour[i][0][1] - contour[i-1][0][1]
                    angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)                       
                    if angle < threshold_angle or angle > 180 - threshold_angle:
                        new_contour.append(contour[i-1])        
                return np.array(new_contour)
    
            
            def preprocess_mask(mask):
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
                # Filter out contours based on their movement direction
                vertical_contours = [filter_horizontal_movements(cnt) for cnt in contours]
            
                # Draw the vertical contours on a new mask
                mask = np.zeros_like(mask)
                for cnt in vertical_contours:
                    # Only draw the contour if it's not empty
                    if cnt.size > 0:
                        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                       
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))    
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) 
                mask = cv2.erode(mask, kernel2, iterations=1)  # Erode the mask to reduce small areas
                mask = cv2.dilate(mask, kernel1, iterations=1)  # Dilate the mask to enhance the vertical contours 
                mask = cv2.erode(mask, kernel1, iterations=1)  # Erode the mask to reduce small areas
                mask = cv2.dilate(mask, kernel1, iterations=1)  # Dilate the mask to enhance the vertical contours 
                return mask
            
            
            def append_json_with_target_data(image):                                               
                image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                img_semseg_bgra = apply_cityscapes_palette(image_data, image.height, image.width)
                img_semseg_rgb = cv2.cvtColor(img_semseg_bgra, cv2.COLOR_BGRA2RGB)            
                cv2.imwrite(Dir+'/output/SEG/S_%.6d.jpg'% image.frame, img_semseg_rgb,[int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                                 
    
                # Get boolean mask and bounding boxes
                img_semseg_hsv = cv2.cvtColor(img_semseg_rgb, cv2.COLOR_RGB2HSV)
                mask = get_hsv_mask(img_semseg_hsv, object_list['RoadLine'])
                mask_greyscale=get_binary_greyscale_mask_from_hsv(mask)
                 
                # Pre-process imags         
                mask_greyscale=preprocess_mask(mask_greyscale)  
                          
                bboxes = get_bboxes_from_mask(mask_greyscale)
                cv2.imwrite(Dir+'/output/MASK/M_%.6d.jpg'% image.frame, mask_greyscale)
                             
                # Create COCO format annotation data            
                annotation_data = create_coco_annotation(image, bboxes, mask_greyscale)
                
                # Append coco image name list
                global coco_data
                coco_data["images"].append({
                    "id": image.frame,
                    "width": image.width,
                    "height": image.height,
                    "file_name": "LF_%.6d.jpg"% image.frame,   #f"LF_{image.frame}.jpg", 
                    "license": 1,
                    "date_captured": "",  # Add date captured information if available
                    "weather_idx": str(weather_idx),
                    "town_idx":str(town_idx),
                })
                
                # Append annotations data to the COCO dictionary (masks polygon and image bounding boxes)
                coco_data["annotations"].extend(annotation_data) #extend used to add elements from an iterable to the end of a list. eg extend list with two dict with a list of two dict gives list of 4 dict
            
            
            def save_segmentation_image(image,file_name):
                #global last_image_frame
                #last_image_frame=copy.deepcopy(image.frame)
                
                # save segmented image to disk using built in carla save method
                #image.save_to_disk(file_name, carla.ColorConverter.CityScapesPalette)
                            
                # save the anotation data
                append_json_with_target_data(image)
                        
                   
            # --------------
            # Add a Segmentation camera to the vehicle. 
            # --------------
            sem_cam = None
            sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            sem_bp.set_attribute("image_size_x",str(640))
            sem_bp.set_attribute("image_size_y",str(480))
            sem_bp.set_attribute("fov",str(105))
            sem_location = carla.Location(1.32,-0.1,1.3)
            sem_rotation = carla.Rotation(0,0,0)
            sem_transform = carla.Transform(sem_location,sem_rotation)
            sem_cam = world.spawn_actor(sem_bp,sem_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            sem_cam.listen(lambda image: save_segmentation_image(image,Dir+'/output/SEG/S_%.6d.jpg' % image.frame))
            #sem_cam.listen(lambda image: image.save_to_disk(Dir+'/output/SEG/S_%.6d.jpg' % image.frame, carla.ColorConverter.CityScapesPalette))
            
    
    
            # --------------
            # Test image 
            # --------------
            '''
            img_semseg_bgra = mpimg.imread(Dir+'/output/SEG/S_319611.jpg' )
            img_semseg_bgr = cv2.cvtColor(img_semseg_bgra, cv2.COLOR_BGRA2BGR)
            img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV)
            #cv2.imwrite(file_name, image_rgb,[int(cv2.IMWRITE_JPEG_QUALITY), 95])
                       
            mask = get_boolean_mask(img_semseg_hsv, object_list['RoadLine'])
            bboxes = get_bboxes_from_mask(mask)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,18))
            ax1.imshow(mask)
            for bbox in bboxes:
                minr, minc, maxr, maxc = bbox
                cv2.rectangle(img_semseg_bgr, (minc,minr), (maxc, maxr), (255,255,255), 6)
            
            ax2.imshow(img_semseg_bgr)
            plt.show()
            
            import pdb; pdb.set_trace()    
            '''
                
        
            # --------------
            # Add collision sensor to ego vehicle. 
            # --------------
            """
            col_bp = world.get_blueprint_library().find('sensor.other.collision')
            col_location = carla.Location(0,0,0)
            col_rotation = carla.Rotation(0,0,0)
            col_transform = carla.Transform(col_location,col_rotation)
            ego_col = world.spawn_actor(col_bp,col_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            def col_callback(colli):
                print("Collision detected:\n"+str(colli)+'\n')
            ego_col.listen(lambda colli: col_callback(colli))
            """
    
            # --------------
            # Add Lane invasion sensor to ego vehicle. 
            # --------------
            
            lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            lane_location = carla.Location(0,0,0)
            lane_rotation = carla.Rotation(0,0,0)
            lane_transform = carla.Transform(lane_location,lane_rotation)
            ego_lane = world.spawn_actor(lane_bp,lane_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            def lane_callback(lane):
                print("Lane invasion detected:\n"+str(lane.frame)+'\n')
                global lane_invasion_frame
                lane_invasion_frame=lane.frame
                
            ego_lane.listen(lambda lane: lane_callback(lane))
            
    
            # --------------
            # Add Obstacle sensor to ego vehicle. 
            # --------------
            """
            obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
            obs_bp.set_attribute("only_dynamics",str(True))
            obs_location = carla.Location(0,0,0)
            obs_rotation = carla.Rotation(0,0,0)
            obs_transform = carla.Transform(obs_location,obs_rotation)
            ego_obs = world.spawn_actor(obs_bp,obs_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            def obs_callback(obs):
                print("Obstacle detected:\n"+str(obs)+'\n')
            ego_obs.listen(lambda obs: obs_callback(obs))
            """
    
            # --------------
            # Add GNSS sensor to ego vehicle. 
            # --------------
            """
            gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
            gnss_location = carla.Location(0,0,0)
            gnss_rotation = carla.Rotation(0,0,0)
            gnss_transform = carla.Transform(gnss_location,gnss_rotation)
            gnss_bp.set_attribute("sensor_tick",str(3.0))
            ego_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            def gnss_callback(gnss):
                print("GNSS measure:\n"+str(gnss)+'\n')
            ego_gnss.listen(lambda gnss: gnss_callback(gnss))
            """
    
            # --------------
            # Add IMU sensor to ego vehicle. 
            # --------------
            """
            imu_bp = world.get_blueprint_library().find('sensor.other.imu')
            imu_location = carla.Location(0,0,0)
            imu_rotation = carla.Rotation(0,0,0)
            imu_transform = carla.Transform(imu_location,imu_rotation)
            imu_bp.set_attribute("sensor_tick",str(3.0))
            ego_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            def imu_callback(imu):
                print("IMU measure:\n"+str(imu)+'\n')
            ego_imu.listen(lambda imu: imu_callback(imu))
            """
    
            # --------------
            # Place spectator on ego vehicle
            # --------------        
            
            #Place spectator camera behind ego vehicle
            camera_bp_rear = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp_rear.set_attribute('image_size_x', '800')
            camera_bp_rear.set_attribute('image_size_y', '600')
            camera_rear_transform = carla.Transform(carla.Location(x=-5.0, z=2.5), carla.Rotation(pitch=-0, yaw=0, roll=0.0))
            camera_rear = world.spawn_actor(camera_bp_rear, camera_rear_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
            image_queue_rear = queue.Queue()
            camera_rear.listen(image_queue_rear.put)               
            #world.get_spectator().set_transform(camera_rear.get_transform()) # Set the spectator camera to rear camera
            return [ego_vehicle,traffic_manager,LF_cam,RF_cam,depth_cam,sem_cam,ego_lane,camera_rear]
        

        print('Spawning new world')
        [ego_vehicle,traffic_manager,LF_cam,RF_cam,depth_cam,sem_cam,ego_lane,camera_rear]=spown_new_world(world,client)
        #import pdb;pdb.set_trace()
        # --------------
        # Main Game loop
        # --------------
        print('Entering game mainloop')
        
        Num_images=30000
                           
        idx_list_wheater=generate_wheather_samples(Num_images) 
        set_weather_by_index(idx_list_wheater[0], world) #MAX_IDX=12  
                          
        count_images=0
        img_dir1=Dir+'/output/Mask'
        img_dir2=Dir+'/output/SEG'
        t1=tm.time()        
        while True:      
            count_images=count_images+1
                       
            if count_images>2 and count_images<Num_images:
                
                # Change town-deletes allold actors (Except trafic manager)
                if count_images % 1000 == 0: #if mod 50 true:     
                    t_ct1=tm.time()                                  
                    try:
                        for actor in [ego_vehicle, LF_cam, RF_cam, depth_cam, sem_cam, ego_lane, camera_rear]:
                            if actor is not None and actor.is_alive:
                                actor.destroy()
                    except Exception as e:
                        print(f"Error occurred while destroying actors: {e}")

                        
                    town_idx=np.random.randint(0,5)
                    set_town_by_index(town_idx, world,client)  # 0 to MAX_IDX=7 
                    [ego_vehicle,traffic_manager,LF_cam,RF_cam,depth_cam,sem_cam,ego_lane,camera_rear]=spown_new_world(world,client)
                    print('time to change town:',tm.time()-t_ct1)
                
                # Chang vehicle spawn point every 50 frames
                if count_images % 50 == 0: #if mod 50 true:
                    spawn_points = world.get_map().get_spawn_points()
                    attempts = 0
                    while attempts < 10: # attemts to replace vehicle if colision on spawn
                        spawn_point = random.choice(spawn_points)
                        try:
                            ego_vehicle.set_transform(spawn_point)                           
                            break
                        except:
                            attempts += 1
                            continue
                                                                
                #change weateher every 10th image  
                if idx_list_wheater[count_images] != idx_list_wheater[count_images-1] and count_images % 5==0:                   
                    weather_idx=idx_list_wheater[count_images]                                          
                    set_weather_by_index(weather_idx, world) # 0 to MAX_IDX=12   
                
               # Update dormant NPC vehicles  every 100 images
                if idx_list_wheater[count_images] != idx_list_wheater[count_images-1] and count_images % 100==0:                      
                    traffic_manager.set_respawn_dormant_vehicles(True) 
                    traffic_manager.set_boundaries_respawn_dormant_vehicles(50,700) 
                    #world.apply_settings(settings)
                 
                    
            # Run simulation and capture images
            world.tick()
            
            # Update the spectator camera view to rear camera
            world.get_spectator().set_transform(camera_rear.get_transform())
            
            # only progress simulation if done writing image files, Progress as fast as possible 
            if tm.time()-t1<=5:
                tm.sleep(5)
            else:                                                
                while 1==1:
                    if (len(os.listdir(img_dir1)) >= count_images) and (len(os.listdir(img_dir2)) >= count_images):
                        break                        
                    else:
                        tm.sleep(0.05)
                        
                        
                        
                
            # break main loop if the specifed number of images have been collected
            if count_images>=Num_images:
                print('Simulation time:',tm.time()-t1-5 )
                print('fps:',  1.0/(  (tm.time()-t1-5)  /(Num_images-1))  )
                
                #fianl close
                settings = world.get_settings()
                settings.synchronous_mode = False
                traffic_manager.set_synchronous_mode(False)
                
                raise Exception("Done colecting images") 
                
    finally:     
        # --------------
        # Save COCO JSON file
        # --------------        
        coco_json_path = os.path.join(Dir, "output/lane_coco_annotations.json")
        
        # save json coco data
        with open(coco_json_path, 'w') as outfile:
            json.dump(coco_data, outfile)

        # load json coco data
        with open(coco_json_path, 'r') as f:
            data = json.load(f)
        
        
        # --------------
        # Stop recording and destroy actors
        # --------------
        # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
        
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            ego_vehicle.destroy()
        
        
            
        #kill carla-Needs admin privalags
        subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=carla_pid))
          
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')