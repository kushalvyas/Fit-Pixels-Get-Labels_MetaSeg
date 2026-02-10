### Setup for MetaSeg 3D segmentation.

 1. Please run step1.ipynb: to generate meta learned initialization for the INR
2. Please run step2.ipynb with `SAVE_FEATURE_VECS = True`. This will generate INR feature vectors for train, val, test sets and store them into './dumps/intermediate_vectors'. This is done because its computationaly simplicity. 
3. Re-run step2.ipynb with `SAVE_FEATURE_VECS = False` . This will start finetuning the segmentation head of metaseg using these saved feature vectors. Please note that thes training segmentation head for 3D data is a tiring and long process and reaching converegence in a day or so approximately. To get best results, use smaller learning rates (>= 5e-5) and let optimization run for longer. Depending on yuor data, you may have to adjust the $\gamma$ parameter in Focal loss. Please read how $\gamma$ controls the focal loss properties in the "Focal Loss for Dense Object Detection by Lin et.al, ICCV 2017". 

4. Once your model is trained, you can run the inference codes. This will directly test on the test-set feature-vectors saved as output of `step2.ipynb`. This script does not render the 3D point clouds. 

5. Incase you want to render 3D point clouds, please fit the signal using the initialization obtained from `step1.ipynb`. 

__Weights from the paper are present in ./weights/ directory. You can directly load them in the inference script.__

