# SOMSOM
Tangerine Orange Flavor Classification //Project of HW design - TAIST-Tokyo Tech
Tangerine orange is a native species of Thailand. It can be used to be an ingredient in many food products. The way to select the delicious tangerine orange is quite difficult because it has to squeeze on the products that make it bruised or worst spoiled. Selecting and separate the flavor also relies on the many experiences and knowledge. So, the alternative way is using the camera with AI to detect an orange image, the color and identity of orange that can tell this orange are sweet or sour. This is very useful when used to sort out the taste of orange in the food industry, in order to allow consumers to eat the desired flavor.👩🏻‍🍳 🍊

# DATASET
The dataset created by ourselves (the creator of this project). We work on 117 oranges and create a dataset by taking a photo of 4 sides of oranges on white background, so the dataset has 468 images of tangerine oranges.

# METHOD
For the processing part, using autoML platform by Google to create detection and classification model. First, get input images from camera has a size of 480 x 640 pixels then resize it to 320 x 320 pixels. After that use resized image to use for detection. In detection model, need to get boundary of each orange. Then do orange segmentation by resize image into 224 x 224 pixels. Next, segmented image will use for flavor classification. After that system will show result of prediction flavor as “SWEET” or “SOUR”

# HARDWARE
 - Coral Dev Board by Google
 - Camera : Logitech HD Webcam c525
 - Monitor Display

# TESTING SYSTEM
After finish build hardware, upload all code and model. We test our system with 26 oranges. Testing step as follow
 - Put orange into the middle of the device
 - Waiting around 1~3 second, result will show up on the monitor screen

# Results
We used 26 tangerine oranges for system testing and create a confusion matrix as in the picture below. From the confusion matrix, we can see that,
 - Have only sweet oranges the accuracy of the prediction system was 87.5%.
 - Have only sour oranges the accuracy of the prediction system was 40%.
 - Oranges that the system predicts as “SWEET”, actual sweet oranges 70%.
 - Oranges that the system predicts as “SOUR”, actual sour oranges 66.7%.
From all of these, the accuracy of the system was 69.2%. So we can see that, when we select only orange that system predicts as “SWEET”, we can get sweet orange around 70% of all oranges.

# AUTHORS
 - Palawich Giraruchataporn 6322040335 m6322040335@g.siit.tu.ac.th
 - Warissara Limpornchitwilai 6322040350 m6322040350@g.siit.tu.ac.th
 - Thanaphon Rianthong 6322040418 m6322040418@g.siit.tu.ac.th

This project is part of ICT730 Hardware Designs for Ai & IoT TAIST-Tokyo Tech program
