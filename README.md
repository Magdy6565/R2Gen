# steps to make inference 

1) Download  Model --> https://drive.google.com/file/d/1fQXpf4vz5t2QYQ89iRXv0U_OVA0MNCtJ/view

2) Clone repo

3) cd R2Gen 

4) !python inference.py --model_path /kaggle/working/model_jordy_bisho.pth  --tokenizer_path tokenizer.pkl --image_path /kaggle/input/chest-xrays-indiana-university/images/images_normalized/1007_IM-0008-3001.dcm.png --dataset_name iu_xray  

# Parameters

--image_path is the path to the Image we want to make inference 


--tokenizer_path path to the tokenizer.pkl file in the repo 


--model_path path to the model you donwloaded in step 1 


--dataset_name Must be iu_xray 

