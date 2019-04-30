# Thermal To Visible Registration
Install the dependencies
- opencv
- numpy
- imutils

# folders to be created to get the output images
- output_orb
- output_sift
- output_hough
- results_orb
- results_sift
- results_hough
- registration_orb
- registration_sift
- registration_hough
- uncropped_orb
- uncropped_sift
- uncropped_hough
- thermal
- visible
# How To Run ?
- Add the thermal and it's corresponding visible image in the foler `thermal` and `visible` folder respectively in the same order, 1st image in visible folder should be the corresponding thermal image of the 1st image in the thermal folder
- make sure you have created the mentioned folders
- To run the registration using ORB feature matching run `python orb.py`. The results will be saved in the folders with _orb
- To run the registration using SIFT feature matching run `python sift.py`. The results will be saved in the folders with _sift
- To run the registration using HOUGH feature matching run `python hough.py`. The results will be saved in the folders with _hough
- The folders named results and uncropped are the final outputs after registration.

To know more about how the project works go through this [article](https://medium.com/@aswinvb/how-to-perform-thermal-to-visible-image-registration-c18a34894866)
