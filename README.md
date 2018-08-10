# Deeplab_V3 gluon Image Semantic Segmentation Network

Implementation of the Semantic Segmentation deepLab_V3 as described in paper Rethinking Atrous Convolution for Semantic Image Segmentation.

Meanwhile,our code incude Vortex pooling,which is described in paper Vortex Pooling: Improving Context Representation in Semantic Segmentation.

## Dependencies:

Python 3.6

Numpy

MXNet>=1.2.0

gluoncv==0.3.0

## Downloads:

#### Dataset(VOC and VOCaug for semantic segmentation)

​    git clone https://github.com/dmlc/gluon-cv

​    cd gluon-cv/scripts/datasets

​    python pascal_voc.py

Then Place the dataset folder inside ./VOCdevkit_aug. If the folder does not exist, create it.

#### Pre-trained model Resnet50_v2

​    Download the Pre-trained model of Resnet50_v2 from  https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet50_v2-81a4e66a.zip, 

​    Unzip it and change it's name to resnet50_v2.params. Then Place the pre-trained model in the main folder(./) .

## Trainning and Eval

​    Once you have the dataset . Just run the shell sh_deeplabv3_train.sh or use the command below.

train.py --train_epochs 46 --resume -1 --batch_size 16 --data_dir ./VOCdevkit_aug --base_architecture resnet_v2_50 --pre_trained_model resnet_v2_50.params --output_stride 16 --freeze_batch_norm 0 --initial_learning_rate 7e-3 --weight_decay 2e-4 --gpus 0,1 --max_iter 30000 --aspp_or_vortex 1

​     Follow the paper's description, for the first 30K iterator , set the batch size =16,output_stride=16 and freeze batch_norm=0(do not freeze batch_norm);for the last 30K iterator,set the batch size =8,output_stride=8 and freeze batch_norm=1 to freeze the batch norm parameters.

## Note:

#### data.py: 

Include (1)The class of VOCSegDatase, which provide data used in DeeplabV3 training.

​              (2)SoftmaxCrossEntropyloss used in training.

#### train.py:

Include (1)Trainning for the whole DeeplabV3;

​              (2)The calculate fo  train and val ’s pix accuracy and mean_iou (they are copy form gluoncv)  

#### resnet/network.py:

Include the whole network architecture of DeeolabV3. 

Meanwhile,our code incude Vortex pooling(the improvement of ASPP in DeepLabV3) ,which is described in paper Vortex Pooling: Improving Context Representation in Semantic Segmentation.You can set  command (or sh_deeplabv3_train.sh) --aspp_or_vortex 1 to use ASPP or set --aspp_or_vortex 2 to use Vortex Pooling.

## Result:

(1) Just use the first 30K iterator.

​    **Pixel accuracy:**   Train:~95%, Val:~95%

​    **Mean Intersection over Union(mean Iou):**   Train:~83%, Val:~73%

(2)add the last 30K iterator.

​     **Pixel accuracy:**   Train:~96%, Val:~96%    

​     **Mean Intersection over Union(mean Iou):**   Train:~83%, Val:~77%



