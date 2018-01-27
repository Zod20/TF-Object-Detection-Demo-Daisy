# Tensorfow Object Detection Training Demo using Daisies

First make sure you install tensorflow on your system. I highly recommend spending the extra time to getting the GPU version running by installing CUDA etc. This will make your training life much easier if you have the GPU power. Install tensorflow by following the instructions here - 

https://www.tensorflow.org/install/

Next you need to download the added examples and models that contain the Tensorflow Object Detection API. I recommend cloning this in your Documents folder. Either go to github and download then zip or clone directly using git (if you have git installed) with the following command - 

     git clone https://github.com/tensorflow/models.git

Next you need to make sure to install all the prerequisites for the object detection api. Follow the installation instructions here - 

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

With that out of the way, your environment is ready and you are finally able to use tensorflow to train!

For this example we will take 10 daisy pictures, hand-label them then train on tensorflow to generate a ssd mobilenet model that can be used on the android detector demo.

Download 10 random daisy jpgs from Google and then arrange them to a folder. You need to have the images named logically, for example 'daisy_1.jpg'. This format is hard-coded in the later scripts so I highly recommend sticking to this format. To automate the naming process, change directory to the folder containing the 10 images and run the following command from bash - 

     ls | cat -n | while read n f; do mv "$f" "Daisy_$n.jpg"; done 

For other images such as for dog images, obviously replace the 'Daisy' in the above command with 'Dog'.


Now that our image have been named automatically, we need to hand-label the coordinates. There's a handy tool that makes this much easier labelImg. 
https://github.com/tzutalin/labelImg

Install labelImg and then use it to hand-label the 10 daisy images you downloaded previously. Set the label as 'Daisy'. You will find that the annotations containing the coordinates are saved with the same file-name as the jpg except with .xml extensions. This is important.

Cool, now we have to convert these images+annotations into TRFrecord format since tensorflow only understands this format. The folder structure is very important for the training to work successfully so please follow the structure I'm writing. First create a folder called 'Daisy_TRFrecord'. Inside create a folder called 'images' and another folder called 'annotations'. Inside 'annotations' create another folder called 'xmls'. I know this is very confusing but that's the way these scripts are hard-coded. Paste the 'create_pet_tf_record.py' into 'Daisy_TRFrecord' folder then create a labelmap as seen on my github (link) which should contain the classes you are training. In our case only daisy.

	Folder Structure - 
	+Daisy_TRFrecord
	 -Daisy_label_map.pbtxt
	 -create_pet_tf_record.py
	 +images
	 +annotations
	  +xmls		
 
 

Paste your 10 images in the 'images' folder and your 10 xml files created from labelImg in the 'xmls' folder. Now you need a 'trainval.txt' which will contain some of the names of the images to be trained. I have a script for this that automates it. Make a backup of the 'img' folder then delete around 70% of the images inside. So in this case delete 7 images. Then from the 'Daisy_TRFrecord' directory run the following command - 

        ls images | grep ".jpg" | sed s/.jpg// > annotations/trainval.txt

This will create the trainval.txt with the remaining images. Now restore the 70% of the images you deleted before so you have all 10 of the original images back in the 'img' folder. With that finally we are done and can generate the TRFrecord file.
//From /Daisy_TRFrecord/ directory

        sudo python create_pet_tf_record.py --label_map_path=Daisy_label_map.pbtxt --data_dir=`pwd` --output_dir=`pwd`

This will create two files - 'pet_train.record' & 'pet_val.record'

We are done with the work in this folder so change directory out and create a new folder called 'Daisy_Train'. Inside create two folders - 'data' & 'models'. Inside 'models' create another two folders 'eval' & 'train'. Paste the 'pet_train.record' & 'pet_val.record' generated from previous step into 'data' folder. Also from my github (link) or elsewhere paste the following scripts inside 'Daisy_Train', 'eval.py' , 'train.py', 'export_inference_graph.py'. Also paste the labelmap you created in the previous step - 'Daisy_label_map.pbtxt' into this folder again. Finally we need two more things for the whole thing to work and that is a config file containing all the training parameters and a pretrained checkpoint to make life easier. The config file can be got from object_detection/samples/configs where you will find lots of training configs. For Android select the 'ssd_mobilenet_v1_pets.config'. You can also find this on my github (link).

Paste this config into 'Daisy_Train' folder then open it and make changes according to the top instructions. Example given below on the lines that should be changed - 

	fine_tune_checkpoint: "models/model.ckpt"
	train_input_reader: {
	  tf_record_input_reader {
	    input_path: "data/pet_train.record"
	  }
	  label_map_path: "/Daisy_label_map.pbtxt"
	}
	eval_input_reader: {
	  tf_record_input_reader {
	    input_path: "data/pet_val.record"
	  }
	  label_map_path: "/Daisy_label_map.pbtxt"
	  shuffle: false
	  num_readers: 1
	}

With that done, paste this modified config file in 'Daisy_Train'. Next we need pretrained checkpoints. To download this, go here and download the appropriate model - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

In our case, since we want android model, download the ssd_mobilenet_v1_coco. Inside you will find 3 files called 'model.ckpt.data-00000-of-00001' , 'model.ckpy.index' , 'model.cpt.meta'
Paste all 3 files into 'Daisy_Train/models/' directory. So our folder structure will finally look like this - 

	Folder Structure (where + means a folder and - means not a folder) - 
	+Daisy_Train
	 -eval.py
	 -train.py
	 -export_inference_graph.py
	 -Daisy_label_map.pbtxt
	 -ssd_mobilenet_v1_pets.config
	 +data
	  -pet_train.record
	  -pet_val.record
	 +models
	  -model.ckpt.data-00000-of-00001
	  -model.ckpy.index
	  -model.cpt.meta
	  +train
	  +eval
 

If you managed to do it exactly like above then congrats, you are nearly done. You can also see the directory structure on my github (link). Now change directory to 'Daisy_Train' and run the following command - 


          sudo python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=ssd_mobilenet_v1_pets.config 

This will start the training process and depending on the number of steps you have selected in your 'ssd_mobilenet_v1_pets.config', it will run that many times. Default setting is 200,000 global steps.

The next step is to start another eval job that will test out the training being carried out. Now the default eval.py never worked for me as the label_map path is missing. So I created a custom custom_eval_ZOD.py that has the label map categories hard coded in. You should edit this file to append the ids and names of all the things you are working on. In this case only id : 1 & name: 'Daisy'.

Once this is done open a new terminal and run the following command from 'Daisy_Train' directory - 


		sudo python custom_eval_ZOD.py --logtostderr --checkpoint_dir=models/train/ --pipeline_config_path=ssd_mobilenet_v1_pets.config --eval_dir=models/eval/

This will start the eval job. Now you can visualize the trainining and eval job by opening another terminal (altogether 3 terminals now) and then changing directory to 'Daisy_Train' then typing the following - 

          tensorboard --logdir=models/

Follow the link shown in a browser to visualize what is happening. If 200,000 steps were selected then it can take around 6-8 hours if you have a strong GPU(I have a 965M). Following which you can now export the model generated to test on the android device. From the 'Daisy_Train' directory run the following command - 

           sudo python export_inference_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix models/train/model.ckpt-200000 --output_directory output_daisy_200000.pb

Obviously if you did let's say 150,000 steps then you would write model.ckpt-150000 in the above command. This will create a 'output_daisy_200000.pb' in your 'Daisy_Train' folder. Inside you will find 'frozen_inference_graph.pb'. Copy this and put it in your android application asset folder (link for the android app) and voila you can now detect daisies on your android app!

Well not really, this example only used 10 pictures and 2000 steps but if you had maybe 1000 pictures and trained for 200000 steps then your application would be able to detect.

Hope this will help someone! :) Any questions then please do not hesitate to ask.
