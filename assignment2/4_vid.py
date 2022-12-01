#!/usr/bin/env python3

import cv2
import depthai as dai
import time
from depthai_sdk.fps import FPSHandler
# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1080,720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)
# Connect to device and start pipeline
start_time = time.time()
x = 1
counter = 0
count=0
imagesf=[]

with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    
    while True:
        videoIn = video.get()
        Frame=videoIn.getCvFrame()
        counter+=1
        cv2.imshow("video", Frame)
        if cv2.waitKey(1) == ord('i'):
            imagesf.append(Frame)
            counter+=1
            
        if cv2.waitKey(1) == ord('p'):
            print('Images have been stitched to make a panaroma')

            if counter < 2:
                print('Not enough pictures to create a panaroma')
            else:
                stitcher=cv2.Stitcher.create()
                image,panaromaview =stitcher.stitch(imagesf)
                if image != cv2.STITCHER_OK:
                    print("could not stitch the images to create a panaroma")
                else:
                    print('Images stitched. Yayyyy Panaroma.')
                    cv2.imshow('Panaroma of the images clicked',panaromaview)
                    cv2.imwrite('panaroma.jpg', panaromaview)

        if cv2.waitKey(1) == ord('q'):
            break