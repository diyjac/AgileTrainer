##
## This is an initial proof-of-concept to interface with the drive.py interface with the Unity SDC simulator
## This was the final outcome from the proof-of-concept, a continuous trainer that interfaces directly with
## the simulator.  Your joystick movement is sent to the simulator and captured by the trainer training the model
## and then the trainer trains the model in batches in the background with your training data randomized, in real-time.
## Even after you finish driving the trainer continues to train the model on your training data.  We are using the
## Adam optimizer with a VERY low learning rate, so it NEVER converges, but the weights the model generates WORKS!
##

## Import some useful modules
import argparse
import base64
import json
import cv2
import pygame
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import Adam
import os
from pathlib import Path
from threading import Thread
from time import sleep
from numpy.random import random

### initialize pygame and joystick
### modify for keyboard starting here!
img_rows, img_cols, ch = 160, 320, 3
pygame.init()
pygame.joystick.init()
size = (img_cols, img_rows)
pygame.display.set_caption("Udacity SDC Project 3: camera video viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
sim_img = pygame.surface.Surface((img_cols,img_rows),0,24).convert()

### PyGame screen diag output
# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

X = []
Y = []

def perspective_tform(x, y):
    p1, p2 = tform3_img((x,y))[0]
    return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, shift_from_mid, sz=1):
    col, row = perspective_tform(x, y)
    row = int(row) + shift_from_mid
    col = int((col+img.get_height()*2)/3)
    if row >= 0 and row < img.get_width()-sz and\
       col >= 0 and col < img.get_height()-sz:
        img.set_at((row-sz,col-sz), color)
        img.set_at((row-sz,col), color)
        img.set_at((row-sz,col+sz), color)
        img.set_at((row,col-sz), color)
        img.set_at((row,col), color)
        img.set_at((row,col+sz), color)
        img.set_at((row+sz,col-sz), color)
        img.set_at((row+sz,col), color)
        img.set_at((row+sz,col+sz), color)

def draw_path(img, path_x, path_y, color, shift_from_mid):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color, shift_from_mid)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi/180.
    slip_fator = 0.0014 # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
    return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
    return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255), shift_from_mid=0):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color, shift_from_mid)

### Preprocessing...
def preprocess(image):
    ## Preprocessing steps for your model done here:
    ## TODO:  Update if you need preprocessing!
    return image

### drive.py initialization
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_prep = np.asarray(image)
    image_array = preprocess(image_prep)

    ### recording flag is replaced with training flag to start image data collection
    training = False
    for event in pygame.event.get():
        continue
        #if event.type == pygame.JOYAXISMOTION:
        #    print("Joystick moved")
        #if event.type == pygame.JOYBUTTONDOWN:
        #    print("Joystick button pressed.")

    ### Get joystick and initialize
    ### Modify/Add here for keyboard interface
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # We are using PS3 left joystick: so axis (0,1) run in pairs, left/right for 2, up/down for 3
    # Change this if you want to switch to another axis on your joystick!
    # Normally they are centered on (0,0)
    leftright = joystick.get_axis(0)/2.0
    updown = joystick.get_axis(1)
    if leftright < -0.01 or leftright > 0.01:
        if joystick.get_button(0) == 0:
            training = True

    ### experiment with joystick button to perform different training function/augmentation?
    #if joystick.get_button(1) == 1:

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    # steering_angle = float(model.predict(transformed_image_array, batch_size=1)) + leftright
    steering_angle = leftright

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.5 + updown
    # print(steering_angle, throttle)
    send_control(steering_angle, throttle)

    ## give us a machine view of the world
    sim_img = pygame.image.fromstring(image.tobytes(), (img_cols, img_rows), 'RGB')

    draw_path_on(sim_img, 0, -steering_angle*20, (0,0,255), 0)
    screen.blit(sim_img, (0,0))
    pygame.display.flip()

    # fill the training and testing queue
    if len(X) < 100:
        X.append(transformed_image_array)
        Y.append(steering_angle)
    else:
        if len(X) == 100:
            print("Ready for training!")

    if training:
        #print("Training: ")
        #print("Right Stick Left|Right Axis value {:>6.3f}".format(leftright) )
        #print("Right Stick Up|Down Axis value {:>6.3f}".format(updown) )
        X.append(transformed_image_array)
        Y.append(steering_angle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

def batchgen(X, Y):
    while 1:
        i = int(random()*len(X))
        y = Y[i]
        image = X[i]
        y = np.array([[y]])
        image = image.reshape(1, img_cols, img_rows, ch)
        yield image, y


def model_trainer(fileModelJSON):
    print("Model Trainer Thread Starting...")

    fileWeights = fileModelJSON.replace('json', 'h5')
    with open(fileModelJSON, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    model.load_weights(fileWeights)
    print("Loaded model from disk:")
    model.summary()

    # start training loop...
    while 1:
        if len(X) > 100:
            batch_size = 20
            samples_per_epoch = int(len(X)/batch_size)
            val_size = int(samples_per_epoch/10)
            if val_size < 10:
                val_size = 10
            nb_epoch = 100

            history = model.fit_generator(batchgen(X,Y),
                                samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                validation_data=batchgen(X,Y),
                                nb_val_samples=val_size,
                                verbose=1)

            print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
            if Path(fileModelJSON).is_file():
                os.remove(fileModelJSON)
            json_string = model.to_json()
            with open(fileModelJSON,'w' ) as f:
                json.dump(json_string, f)
            if Path(fileWeights).is_file():
                os.remove(fileWeights)
            model.save_weights(fileWeights)
        else:
            print("Not Ready!  Sleeping for 5...")
            sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    # start training thread
    thread = Thread(target = model_trainer, args=(args.model,))
    thread.start()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    # wait for training to end
    thread.join()

