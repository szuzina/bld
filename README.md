# bld
Bidirectional local distance in Python

Link of the Colab notebook ("main"):
(https://colab.research.google.com/drive/1OnJABs1JgkQyQB6p3IpbVhhAFSL36dps?usp=sharing)

Background: 
In the medical field, more and more image data is being created every day. The emerging quantity of data makes the different algorithms more and more important as nowadays it is quite impossible to analyze this large quantity of data by hand. There are several different kinds of automatic or semi-automatic medical image segmentation algorithms, but it is crucial to evaluate their performance in a clinically relevant manner. Several widely used evaluation metrics (Dice and Jaccard index, Hausdorff distance, etc.) are poorly correlated with clinical relevance. These indices may show very good value, but for example, if a crucial organ is close to the segmented area, a minimal overestimation of the lesion may have a huge clinical effect. 

Aim: 
As the widely used metrics do not correlate with the clinical relevance and there is no possibility for adjusting these metrics to the different application domains, our aim was to develop a new segmentation evaluation metric which can modified by the user based on the desired clinical application.

Summary: 
We created a new segmentation evaluation metric which can be adjusted based on the planned clinical aim.
Currently an inside and an outside penalty level can be defined, and the MSI is calculated by the BLD and the previously defined weight function. 
The code at this point works with more than one contour on one slice, howevwe, the aggregation is not defined yet (we get as many MSI values as the number of the contours). However, the get_contour_from_image function returns with more than one list if there is more than one contour on one slice.
