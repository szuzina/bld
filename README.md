# MSI
Implementation of the Medical Similarity Index in Python
(along with nnUNet training pipeline, mask splitting algorithm and visualization graphs)

![Pipeline overview](https://drive.google.com/file/d/1WYq9aOcY86atrh9B0EAAOkhroGucCxS1/view?usp=sharing)


Link of the Colab notebook for MSI calculation:
 <a target="_blank" href="https://colab.research.google.com/drive/1BNvLGiS4pBb3i4InozwbPLZ9j_l1nJvn?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg"
alt="Open In Colab"/>
 </a>
 See data file requirements [here](https://github.com/szuzina/bld/wiki/Data_requirements)

Link of the Colab notebook for visualization:
 <a target="_blank" href="https://colab.research.google.com/drive/1yyOAnRElT7RZt2_VHqXo_HVFNh400Lf1?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg"
alt="Open In Colab"/>
 </a>

Link of the Colab notebook for nnUNet training:
 <a target="_blank" href="https://colab.research.google.com/drive/1J8aW5w5fZGZ66E5MSXc1qaAgbaksmNuZ?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg"
alt="Open In Colab"/>
 </a>
(works for prostate, needs to be adjusted for myomas with nifti padding)

We used `Python` version `3.10`.
The necessary packages and version numbers are listed in `requirements.txt` file.
It should be installed as `pip install -r requirements.txt`.


About the study:

Background: In the field of radiology and radiotherapy, accurate delineation of different tissues and organs plays a crucial role in both diagnostics and therapeutics. While the gold standard remains expert-driven manual segmentation, many machine learning-based automatic segmentation methods are emerging. The evaluation of these methods mainly relies on traditional area-based and distance-based metrics, but these metrics only incorporate geometrical properties and fail to adapt to different clinical applications. Thus, there is an understandable need for a clinically meaningful, reproducible assessment of autocontouring systems. 

Aims: This study aims to develop and implement a clinically relevant segmentation metric that can be adapted to different medical imaging applications.

Methods: The reference contour was considered the gold standard segmentation, and, after pairing the contours, the agreement of a test contour to the reference contour was quantified. The bidirectional local distance was defined, and based on this distance, the points of the test contour were paired to points of the reference contour. After correcting for the distance between the test and reference center of mass, the Euclidean distance was calculated between the paired points, and a score was given to each test point. The overall medical similarity index was calculated as the average scores across all the test points. The fine-tuning of the user-defined hyperparameters was demonstrated with an open-access anatomic prostate segmentation MRI dataset. We trained an nnUNet neural network for segmentation, and manually selected six test cases (two easy, two moderate and two difficult cases) for evaluation.
