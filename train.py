#coding=utf-8
import mxnet as mx
from gluoncv.utils import LRScheduler
import argparse
from data import VOCSegDataset,SoftmaxCrossEntropyLoss
from mxnet import autograd
from resnet.network import ResNet
from resnet.network import BottleneckV2
from mxnet import gluon
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', type=int, default=46,
                    help='Number of training epochs: '
                         'For last 30K iteration with batch size 8, train_epoch = 82.38 (= 30K * 8 / 2913),output_stride=8,trainval has 2913'
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')
parser.add_argument('--resume', type=int, default=-1,
                    help='resume training from epoch n')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of examples per batch.')
parser.add_argument('--data_dir', type=str, default='./VOC2012/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')
parser.add_argument('--base_architecture', type=str, default='resnet_v2_50',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')
parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')
parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')
parser.add_argument('--freeze_batch_norm', type=int,default=0,
                    help='Freeze batch normalization parameters during the training.')
parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')
parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')
parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')
parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')
parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0,1', type=str)
parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')
parser.add_argument('--aspp_or_vortex', type=int, default=1,
                    help='use aspp or vertox,if =1,use aspp,or use vertox.')
args = parser.parse_args()
import mxnet.ndarray as F

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = F.argmax(output, 1)
    predict = predict.asnumpy() + 1
    target = target.asnumpy().astype(predict.dtype) + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = F.argmax(output, 1)
    target = target.astype(predict.dtype)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.asnumpy() + 1
    target = target.asnumpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
   
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def _get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch
    return (gluon.utils.split_and_load(features, ctx),
            gluon.utils.split_and_load(labels, ctx),
            features.shape[0])

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """评价模型在数据集上的准确率。"""
    total_inter, total_union, total_correct, total_label = (0,) * 4
    for i, (x, y) in enumerate(data_iter):
        x = x.copyto(mx.gpu(0))
        #y = y.copyto(mx.gpu())
        pred = net(x).copyto(mx.cpu())
        correct, labeled = batch_pix_accuracy(output=pred, target=y)
        inter, union = batch_intersection_union(output=pred, target=y, nclass=21)
        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        pix_acc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mx.nd.waitall()
        # break
    return pix_acc, mIoU

def train_net(train_epoch,ctx,batch_size,data_dir,pre_trained_model,output_stride, \
              freeze_batch_norm,initial_learning_rate,weight_decay,base_architecture,aspp_or_vortex,resume):

    if base_architecture=='resnet_v2_50':
        print('use resnet_v2_50')
        net = ResNet(BottleneckV2, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], output_stride,aspp_or_vortex)
    elif base_architecture=='resnet_v2_101':
        print('use resnet_v2_101')
        net = ResNet(BottleneckV2, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], output_stride,aspp_or_vortex)   

    if resume>=0:
        print('resume for continue trainning')
        begin_epoch=resume+1
        model_path='./checkpoint/deeplabv3_%s.params'%resume
      
        net.initialize(ctx=ctx)
        print('model_path',model_path)
        net.collect_params().load(model_path,ctx=ctx,restore_prefix='')
        
        if output_stride==8:
            begin_epoch=resume+1-46
        if freeze_batch_norm==1:
            print('In the last 30K iters,freeze batchnorm ')
            net.collect_params('.*gamma|.*beta|.*running_mean|.*running_var').setattr('grad_req', 'null')    
    else:
        print('begin trainning')
        begin_epoch=0
        print('before auto init')
        net.initialize(ctx=ctx)
        print('after auto init')
        net.load_params(pre_trained_model,ctx=ctx,allow_missing=True,ignore_extra=True)
    
    loss = SoftmaxCrossEntropyLoss() 
    #first 30K iter ,use split='train_aug',the last 30K iter use trainval. 
    train_data = VOCSegDataset(root=data_dir, split='trainval')
    val_data = VOCSegDataset(root=data_dir, split='val')
    train_dataiter = gluon.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, last_batch='discard')
    val_dataiter = gluon.data.DataLoader(val_data, batch_size=batch_size, last_batch='discard')  
    lr_scheduler = LRScheduler(mode='poly', baselr=initial_learning_rate, niters=len(train_dataiter),
                                        nepochs=train_epoch)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                       {'lr_scheduler': lr_scheduler,
                                        'wd': weight_decay,
                                        'momentum': 0.9,
                                        'multi_precision': True})
    for epoch in range(begin_epoch,train_epoch):
        train_loss, train_acc,meaniou, n, m = 0, 0, 0, 0,0
        total_inter, total_union, total_correct, total_label = (0,) * 4
        iter=0
        for i,batch in enumerate(train_dataiter):
            data, label, batch_size = _get_batch(batch, ctx)
            lr_scheduler.update(i, epoch)

            with autograd.record():
                output=[net(x) for x in data]
                losses=[loss(yhat,y) for yhat,y in zip(output,label)]
         
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in losses])
            
            n += batch_size
             
            m += sum([y.size for y in label])
            #  evaluation
            correct, labeled=(0,)*2
            result_pix= [batch_pix_accuracy(output_,label_) for
                            output_, label_ in zip(output, label)]
            for i in range(len(result_pix)):
                correct+=result_pix[i][0]
                labeled+=result_pix[i][1]
            inter, union=(0,)*2
            result_iou=[batch_intersection_union(output_, label_,21) for
                    output_, label_ in zip(output, label)]
            for i in range(len(result_iou)):
                inter+=result_iou[i][0]
                union+=result_iou[i][1]
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pix_acc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            iter = iter + 1
            if iter % 10 == 0:
                print('-Epoch %s, Batch %d. Loss: %f, pix_acc: %.4f, mIoU: %.4f' % (epoch,n, train_loss / n, pix_acc, mIoU))
        net.collect_params().save(filename='./checkpoint/deeplabv3_%s.params' % (epoch))

        val_pix_acc, val_mIoU = evaluate_accuracy(val_dataiter, net, ctx)
        print('val_pix_acc: %.4f, val_mIoU: %.4f' % (val_pix_acc, val_mIoU))

def main():
    print ('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args.train_epochs, ctx, args.batch_size,args.data_dir,\
              args.pre_trained_model,args.output_stride,args.freeze_batch_norm,args.initial_learning_rate,\
              args.weight_decay,args.base_architecture,args.aspp_or_vortex,args.resume)
if __name__ == '__main__':
    main()

