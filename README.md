# Road Lane Line Detection


![title](/Images/introduction.PNG)


In this project, I implemented a computer vision algorithm that processes real data recorded with the front-facing camera of a vehicle driving on an Israel, 20 Ayalon North on the way to work.
The result is a processed video that highlights the lane lines on the paved road.

## Project management stages:

 ![title](/Images/frontCamera.PNG)
 
1. Road Mask
2. Color masks	
3. Canny Edge Detection
4. Hough Transformation                  
5. Angle & Location Filter 
6. Draw Line Average


# 
###### Road Mask
Cutting down the dashboard, sky, and all the irrelevant information to reduce unwanted noise.
I created a triangle-shaped mask customized to our needs.

