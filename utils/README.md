
How to use the `utils` folder to train `nnUNet` neural network?

The additional package requirements are listed in the `requirements_nnunet.txt` file, 
it needs to be installed as `pip install -r requirements_nnunet.txt`.

Check if `cuda` is installed and properly working.

`print(torch.cuda.is_available())`

The declaration of the environmental variables is needed.

`export nnUNet_raw="..."`

`export nnUNet_preprocessed="..."`

`export nnUNet_results="..."`

Do the preprocess.

`nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity`

If preprocessed went fine, start training.

`nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD`
(caution: default 1000 epoch per fold)

After getting all the folds, can do the prediction.

`nnUNetv2_predict -d DATASET_ID -i INPUT_FOLDER -o OUTPUT_FOLDER -c CONFIGURATION -f 0 1 2 3 4-`