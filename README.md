# Road Lane Line Detection


![title](/Images/introduction.PNG)

In this project, I implemented a computer vision algorithm that processes real data recorded with the front-facing camera of a vehicle driving on an Israel, 20 Ayalon North on the way to work.
The result is a processed video that highlights the lane lines on the paved road.
###### Video Link
 
- Link to [Youtube](https://www.youtube.com/watch?v=3_SZbX6wxe8&ab_channel=ItayNave).

## Project steps:

 ![title](/Images/frontCamera.PNG)
 
1. Road Mask
2. Color masks	
3. Canny Edge Detection
4. Hough Transformation                  
5. Angle & Location Filter 
6. Draw Line Average

# 
###### Road Mask
> Cutting down the dashboard, sky, and all the irrelevant information to reduce unwanted noise.
> I created a triangle-shaped mask customized to our needs.
>  
> ![title](/Images/triangle_mask.PNG)

###### Color Masks
> Defining color masks allows color-based pixels selection in an image. 
> 
> The intention is to select only orange and yellow pixels and set the rest of the image to black.
> 
> ![title](/Images/color_mask.PNG)

###### Canny Edge Detection
> The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
> to detect a wide range of edges in images.
> ```
> 1.	Apply Gaussian filter to smooth the image to remove the noise
> 2.	Find the intensity gradients of the image
> 3.	Apply non-maximum suppression to get rid of spurious response to edge detection
> 4.	Apply double threshold to determine potential edges
> 5.	Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edge
> ```
> ![title](/Images/canny.PNG)

###### Hough Transformation
> The Hough transform been used to detect the lines and the output is a parametric description of the lines in an image.
> 
> ![title](/Images/Hough-Transformation.PNG)
> 
> We got many lines plus noises, so it is necessary to create an angle filter and draw one average line for a good result.
>

###### Angle and Location Filter
> 
> ![title](/Images/Angle_and_Location_Filter.jpg)
> 
> Sort the lines array into 2 parts by location and vector angle.
>
> | Left Lines      | Right Lines |
> | ----------- | ----------- |
> | __Xmin  <  X1  <  Xmiddle__      |  __Xmiddle  <  X1  <  Xmax__       |
> |  __20ﹾ  <  θ  <  40ﹾ__  |  __-40ﹾ  <  θ  <  -20ﹾ__  |
>
> 

###### Draw Line Average
> Average on the 2 groups separately,
> And printing an average line, right and left.
> 
> ![title](/Images/Draw_line_average.PNG)
>
> 
> * Green Lines: lines after angle filter.
> * Blue Lines: lines after average.
> 



