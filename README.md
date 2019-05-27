# Enhanced Deep Residual Networks for Single Image Super-Resolution

# Requirements
<br>python 2.7
<br>Tensorflow 1.9.0

# Usage
### I have not use data augment to train this code and the PSNR on Set5 is about 37.70+ lower than the result on paper.
### To use this code, you should:
<br>1. run "train_data_generator_div2k.m" to get train data. 
<br>2. run "generate_test.m" to get test data. (This data not the final test!!!)
<br>3. run "train.py" for training network.
<br>4. run "get_test_EDSR.m" to get test data.
<br>5. run "test.py" to test the model.


## Update
### 2019-05-27
The old code use the low_image.shape to calculate the psnr.

Add a new line of code in "test.py"(91 line) to use the original_image.shape.
