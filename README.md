# EDSR_Tensorflow

<br>python 2.7
<br>Tensorflow 1.9.0

### I have not use data augment to train this code and the PSNR on Set5 is about 37.70+ lower than the result on paper.
### To use this code, you should:
<br>1. run "train_data_generator_div2k.m" to get train data. 
<br>2. run "generate_test.m" to get test data. (This data not the final test!!!)
<br>3. run "train.py" for training network.
<br>4. run "get_test_EDSR.m" to get test data.
<br>5. run "test.py" to test the model.


### I find that if use tf.nn.conv2d the performance of this code is very bad. So I use tf.layers.conv2d instead. If you the reason please tell me (never_look_back6@163.com). Many thanks!
