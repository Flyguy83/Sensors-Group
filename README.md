Sensors Project 3 Group 2

This script implements Image-Based Visual Servoing (IBVS) to control a UR3 robotic arm using Intel RealSense camera feedback. 
The goal is to iteratively minimise the pixel error between the observed feature positions in the image and a desired target location.
The checkerboard should begin 80cm away from the RealSense camera, moving after script initialisation

Before running the program, ensure that:
- The Intel RealSense camera is connected to the device
- The host IP and port are correct

When the program is run: 
- A live camera feed window will open
- A 2-second delay will occur before the main loop begins

It will continue to run unless CTRL+C is pressed, during which it will continuously reduce pixel error via end effector movement
