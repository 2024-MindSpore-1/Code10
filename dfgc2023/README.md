# DFGC-VRA

 This code is a MindSpore implementation of the 2nd-place solution of the DeepFake Game Competition on Visual Realism Assessment [(DFGC-VRA)](https://codalab.lisn.upsaclay.fr/competitions/10754) held with IJCB2023.

 We are currently working on a unitive inplementation of all top 5 solutions.

**Note:** I rewrote most of the code to include the implementation of the 1st solution and adopted a new way to arrange the dataset. Please redownload this project if you are using older versions.
Dataset now contains extracted frames at the original size without cropping, and the bounding box for cropping is saved in the ```crop``` folder. Cropping will be processed when loading the dataset. I saved 1 frame for every 6 frames.

Some pretrained parameters and training settings can be found [here](https://drive.google.com/drive/folders/1aQGIZaFAYf-azJ7P6Gt4w20J11-50QaN?usp=sharing), these are parameters converted from the submitted pytorch model of each team. More will be added.

For detailed information of the settings in the ```config``` files, please refer to the comments. You may change their values for better results.
The reproduced results might be lower than the leaderboard results, as I manipulate the implementations a little bit to unite different methods into one framework.