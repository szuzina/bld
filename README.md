# bld
Implementation of the Medical Similarity Index in Pythonű
(along with nnUNet training pipeline, mask splitting algorithm and visualization graphs)

Link of the Colab notebook ("main"):
(https://colab.research.google.com/drive/1OnJABs1JgkQyQB6p3IpbVhhAFSL36dps?usp=sharing)

Background: In the field of radiology and radiotherapy, accurate delineation of different tissues and organs plays a crucial role in both diagnostics and therapeutics. While the gold standard remains expert-driven manual segmentation, many machine learning-based automatic segmentation methods are emerging. The evaluation of these methods mainly relies on traditional area-based and distance-based metrics, but these metrics only incorporate geometrical properties and fail to adapt to different clinical applications. Thus, there is an understandable need for a clinically meaningful, reproducible assessment of autocontouring systems. 

Aims: This study aims to develop and implement a clinically relevant segmentation metric that can be adapted to different medical imaging applications.

Methods: The reference contour was considered the gold standard segmentation, and, after pairing the contours, the agreement of a test contour to the reference contour was quantified. The bidirectional local distance was defined, and based on this distance, the points of the test contour were paired to points of the reference contour. After correcting for the distance between the test and reference center of mass, the Euclidean distance was calculated between the paired points, and a score was given to each test point. The overall medical similarity index was calculated as the average scores across all the test points. The fine-tuning of the user-defined hyperparameters was demonstrated with an open-access anatomic prostate segmentation MRI dataset. We trained an nnUNet neural network for segmentation, and manually selected six test cases (two easy, two moderate and two difficult cases) for evaluation.

Results: An easy-to-use, sustainable image processing pipeline was created using Python. The code is available in this public GitHub repository along with a Google Colaboratory notebook for calculating MSI and traditional segmentation metrics. The algorithm can handle multi-slice images with more than one mask in one slice. Additionally, a mask splitting algorithm is also provided that can separate the concave masks. The Google Colaboratory notebook is also provided for the reproducibility of the nnUNet training. The masks of our own myoma dataset is available, additionally the clinical relevance and adaptability is highlighted with prostate anatomic segmentation evaluation from an open-access dataset.

Conclusion: A novel segmentation evaluation metric was implemented, and an open-access image processing pipeline was also provided, which can be easily used for automatic measurement of clinical relevance of medical image segmentation. The pipeline enables the calculation of MSI and traditional image segmentation metrics and fine-tuning for clinical use. This tool offers a reproducible and adaptable framework for evaluating autocontouring systems in medical imaging.
