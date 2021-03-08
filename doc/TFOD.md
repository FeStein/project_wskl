# TF Object Detection API using Docker

Check that proprietary nvidia dirvers are correctly installed:
```bash
nvidia-smi
```
This command can also be used to check the gpu resources used.

## Structure

* **tfod/** - relevant packages (i.e. src for tf object dection)
* **scripts/** - source code
* **data/** - datasets
* **cfg/** - model configuration

## Verify Installation

```bashc
python3 /tfod/models/research/object_detection/builders/model_builder_tf2_test.py
```
This checks that if all models included in the TFOD API are correctly working.
Expected output:
```
[...]
----------------------------------------------------------------------
Ran 20 tests in 16.772s

OK (skipped=1)
```

## Check that GPU can be loaded correctly

```bash
python3 scripts/test_gpu.py
```
Expected output:
```bash
[loading different tf libraries (INFO log)]
Num GPUs Available:  1
```
Then verify that CUDA is working correctly:

## Check CUDA installation

Verify the CUDA Toolkit version
```bash
nvcc -V
```
Expected Output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
```

# Prepare object detection

## Create Split/Test Dataset

Extract dataset, specify paths and run the script via 
```bash
python3 split_train_test.py
```
The result should be an  **image/** folder containing a **train/** and **test/**
folder with the split dataset (90:10).

## Create Labelmap (Define Labels for all classes)
example:
```
item{
    id: 1
    name: 'Schraubenzieher'
}
item{
    id: 2
    name: 'Hammer'
}
```

## Create Tensorflow Records
Run inside **/data/** of the docker container:
```bash
python3 generate_tfrecord.py -x '/data/images/train' -l '/data/annotations/label_map.pbtxt' -o '/data/annotations/train.record'
python3 generate_tfrecord.py -x '/data/images/test' -l '/data/annotations/label_map.pbtxt' -o '/data/annotations/test.record'
```

## Train the model
Faster R-CNN
```bash
python4 /scripts/model_main_tf2.py --model_dir=/cfg/models/faster_rcnn_v1 --pipeline_config_path=/cfg/models/faster_rcnn_v1/pipeline.config
```
Efficienet
```bash
python3 /scripts/model_main_tf2.py --model_dir=/cfg/models/EfficientDet_D4_v1 --pipeline_config_path=/cfg/models/EfficientDet_D4_v1/pipeline.config
```

## Export the model graph
```bash
python3 /scripts/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /cfg/models/EfficientDet_D4_v1/pipeline.config --trained_checkpoint_dir /cfg/models/EfficientDet_D4_v1/ --output_directory /cfg/exported-models/EfficientDet_D4_v1/
```

## Run detection
```bash
python3 /scripts/run_saved_model.py
```

