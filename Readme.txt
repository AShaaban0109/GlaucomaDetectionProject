2 Jupiter notebooks are provided showing example code:

#################################################################################

The first notebook 'x' provides example code for running a single input model. 
It is important to note that minor changes need to be made to the code to run different 
architectures. The 2 changes that must be made are the following":

1. The following line must be altered to import the pretrained model and its 
	respective preprocess_input function as needed:
	from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

2. The following must be altered to use the pretrainded model selected:
	transfer = InceptionV3(weights='imagenet', include_top=False, 
				input_tensor=model_input)

Note that for 2) there is an addition alpha parameter 'alpha=0.75' that was used in the 
MobileNet v3 model training.

#################################################################################

The second notebook is similar in a lot of ways to the first. The 2 changes made in the first
also need to be made in the second. Additionaly:

3. To select the features to include in the training of the model the following extra_col
	argument must be changed for each of the 3 generators:
	Eg. from	 extra_col= ["sex", "age"] 
	    to		 extra_col= ["sex"]

4. And edit the following to agree with the number of extra inputs you will provide:
	Eg. model = create_multi_input_model(image_size=224, extra_input_count = 1)
		
#################################################################################

Notes:

This project was completed on Kaggle. The ROOT directory can be changed to allow the project 
to be run elsewere:
	ROOT = "/kaggle/input/multichannel-glaucoma-benchmark-dataset"

The notebooks have been quickly loaded and trained for 2 epochs just to enable a quick
visualisation. The user can change the number of epochs to train the model here:
	history = model.fit(trainGen, steps_per_epoch=len(trainGen), validation_data=valGen,
			    validation_steps=len(valGen), epochs=2, callbacks=[reduce_lr])
