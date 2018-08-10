#coding=utf-8
from mxnet.gluon import nn
import mxnet as mx

class BottleneckV2(nn.HybridBlock):
    def __init__(self,i,channels, stride,dilation=1,**kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.channels=channels
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
   
        self.conv2 = nn.Conv2D(channels//4, kernel_size=3, strides=stride, padding=dilation,
                     use_bias=False,dilation=dilation)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        self.stride=stride

        if i==0:
            self.downsample_c = nn.Conv2D(channels, 1, stride, use_bias=False)

        self.downsample_m=nn.MaxPool2D([1,1],strides=stride)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.channels!=x.shape[1]:
            residual = self.downsample_c(x)
        elif self.stride==2:
            residual = self.downsample_m(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual
class atrous_spatial_pyramid_pooling(nn.HybridBlock):
    def __init__(self,output_stride,depth=256,**kwargs):
        super(atrous_spatial_pyramid_pooling, self).__init__(**kwargs)
        with self.name_scope():
            self.GLBpooling=nn.GlobalAvgPool2D()
            self.image_level_features=nn.Conv2D(depth,kernel_size=1)
            self.at_pool1x1=nn.Conv2D(depth,kernel_size=1)
            if output_stride==16:
                self.at_pool3x3_1=nn.Conv2D(depth,kernel_size=3,dilation=6,padding=6)
                self.at_pool3x3_2=nn.Conv2D(depth,kernel_size=3,dilation=12,padding=12)
                self.at_pool3x3_3=nn.Conv2D(depth,kernel_size=3,dilation=18,padding=18)
            elif output_stride==8:
                self.at_pool3x3_1 = nn.Conv2D(depth, kernel_size=3, dilation=12, padding=12)
                self.at_pool3x3_2 = nn.Conv2D(depth, kernel_size=3, dilation=24, padding=24)
                self.at_pool3x3_3 = nn.Conv2D(depth, kernel_size=3, dilation=36, padding=36)
            self.last=nn.Conv2D(depth,kernel_size=1)
            self.bn_image = nn.BatchNorm()
            self.bn_1x1 = nn.BatchNorm()
            self.bn_3x3_1 = nn.BatchNorm()
            self.bn_3x3_2 = nn.BatchNorm()
            self.bn_3x3_3 = nn.BatchNorm()
            self.bn_last=nn.BatchNorm()
    def hybrid_forward(self, F, x):
        image_level_features=self.image_level_features(self.GLBpooling(x))
        image_level_features=self.bn_image(image_level_features)
        at_pool1x1=self.at_pool1x1(x)
        at_pool1x1=self.bn_1x1(at_pool1x1)
        at_pool3x3_1=self.at_pool3x3_1(x)
        at_pool3x3_1=self.bn_3x3_1(at_pool3x3_1)
        at_pool3x3_2=self.at_pool3x3_2(x)
        at_pool3x3_2=self.bn_3x3_2(at_pool3x3_2)
        at_pool3x3_3=self.at_pool3x3_3(x)
        at_pool3x3_3=self.bn_3x3_3(at_pool3x3_3)
        image_level_features=F.contrib.BilinearResize2D(image_level_features,height=x.shape[2],width=x.shape[2])
        out=F.concat(image_level_features,at_pool1x1,at_pool3x3_1,at_pool3x3_2,at_pool3x3_3)
        out=self.last(out)
        out=self.bn_last(out)
        return out

class Vortex_Pooling(nn.HybridBlock):
    def __init__(self,depth=256,**kwargs):
        super(Vortex_Pooling, self).__init__(**kwargs)
        with self.name_scope():
            self.GLBpooling = nn.GlobalAvgPool2D()
            self.image_level_features = nn.Conv2D(depth, kernel_size=1)
            self.at_pool1x1 = nn.Conv2D(depth, kernel_size=1)

            self.at_pool3x3_1 = nn.Conv2D(depth, kernel_size=3, dilation=3, padding=3)
            self.at_pool3x3_2 = nn.Conv2D(depth, kernel_size=3, dilation=9, padding=9)
            self.at_pool3x3_3 = nn.Conv2D(depth, kernel_size=3, dilation=27, padding=27)

            self.last = nn.Conv2D(depth, kernel_size=1)
            self.bn_image = nn.BatchNorm()
            self.bn_1x1 = nn.BatchNorm()
            self.bn_3x3_1 = nn.BatchNorm()
            self.bn_3x3_2 = nn.BatchNorm()
            self.bn_3x3_3 = nn.BatchNorm()
            self.bn_last = nn.BatchNorm()
            #vortex pooling
            self.avg3 = nn.AvgPool2D(pool_size=(3, 3), strides=1, padding=1)
            self.avg9 = nn.AvgPool2D(pool_size=(9, 9), strides=1, padding=4)
            self.avg27 = nn.AvgPool2D(pool_size=(27, 27), strides=1, padding=13)
    def hybrid_forward(self, F, x):
        image_level_features=self.image_level_features(self.GLBpooling(x))
        image_level_features=self.bn_image(image_level_features)
        at_pool1x1=self.at_pool1x1(x)
        at_pool1x1=self.bn_1x1(at_pool1x1)

        avg_3=self.avg3(x)
        at_pool3x3_1=self.at_pool3x3_1(avg_3)
        at_pool3x3_1=self.bn_3x3_1(at_pool3x3_1)

        avg_9=self.avg9(x)
        at_pool3x3_2=self.at_pool3x3_2(avg_9)
        at_pool3x3_2=self.bn_3x3_2(at_pool3x3_2)

        avg_27=self.avg27(x)
        at_pool3x3_3=self.at_pool3x3_3(avg_27)
        at_pool3x3_3=self.bn_3x3_3(at_pool3x3_3)
        image_level_features=F.contrib.BilinearResize2D(image_level_features,height=x.shape[2],width=x.shape[2])
        out=F.concat(image_level_features,at_pool1x1,at_pool3x3_1,at_pool3x3_2,at_pool3x3_3)
        out=self.last(out)
        out=self.bn_last(out)
        return out
class ResNet(nn.HybridBlock):
    def __init__(self,block,layers,channels,output_stride,aspp_or_vortex=1,**kwargs):
        super(ResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels)-1
        self.aspp_or_vortex=aspp_or_vortex
        with self.name_scope():          
            self.features = nn.HybridSequential(prefix='ppss1')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))
           
            if output_stride==16:
                self.features.add(self._make_layer(block, layers[0], channels[1], 0, stride=2, in_channels=channels[0], \
                        output_stride =output_stride))
                self.features.add(self._make_layer(block, layers[1], channels[2], 1, stride=2, in_channels=channels[1], \
                        output_stride=output_stride))
                self.features.add(self._make_layer(block, layers[2], channels[3], 2, stride=1, in_channels=channels[2], \
                        output_stride =output_stride))
                self.features.add(self._make_layer(block, layers[3], channels[4], 3, stride=1, in_channels=channels[3], \
                        output_stride =output_stride))
            elif output_stride==8:
                self.features.add(self._make_layer(block, layers[0], channels[1], 0, stride=2, in_channels=channels[0], \
                        output_stride =output_stride))
                self.features.add(self._make_layer(block, layers[1], channels[2], 1, stride=1, in_channels=channels[1], \
                        output_stride =output_stride))
                self.features.add(self._make_layer(block, layers[2], channels[3], 2, stride=1, in_channels=channels[2], \
                        output_stride=output_stride))
                self.features.add(self._make_layer(block, layers[3], channels[4], 3, stride=1, in_channels=channels[3], \
                        output_stride=output_stride))

            ###assp部分
            self.relu=nn.Activation(activation='relu')
            self.bn=nn.BatchNorm()
            if aspp_or_vortex==1:
                self.assp=atrous_spatial_pyramid_pooling(output_stride)
            if aspp_or_vortex==2:
                self.vortex=Vortex_Pooling()
            self.class_total=nn.Conv2D(channels=21,kernel_size=1)
            self.features.collect_params()
    def _make_layer(self,block,layers,channels,stage_index,stride=1,in_channels=0,output_stride=16):
        #因为模型第一个就是stage1，不是stage0，所以只能先写成prefix=stage_index+1
        layer = nn.HybridSequential(prefix='stage%d_' % (stage_index+1))
        with layer.name_scope():
            if output_stride == 16:
                if stage_index != 3:
                    for _ in range(layers - 1):
                        layer.add(block(_,channels, 1, dilation=1, prefix=''))
                    layer.add(block(1,channels, stride, dilation=1, prefix=''))
                else:
                    layer.add(block(0,channels, stride, dilation=2, prefix=''),
                              block(1,channels, stride, dilation=4, prefix=''),
                              block(1,channels, stride, dilation=8, prefix=''))
            elif output_stride == 8:
                if stage_index != 2:
                    for _ in range(layers - 1):
                        layer.add(block(_,channels, 1, dilation=1, prefix=''))
                    layer.add(block(1,channels, stride, dilation=1, prefix=''))
                elif stage_index==2:
                    for _ in range(layers):
                        layer.add(block(_,channels, 1, dilation=2, prefix=''))
                elif stage_index==3:
                    layer.add(block(0,channels, stride, dilation=4, prefix=''),
                              block(1,channels, stride, dilation=8, prefix=''),
                              block(1,channels, stride, dilation=16, prefix=''))
        layer.collect_params()
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.bn(x)
        x=self.relu(x)
        if self.aspp_or_vortex==1:
            x=self.assp(x)
        if self.aspp_or_vortex==2:
            x=self.vortex(x)
        x=self.class_total(x)
        x = F.contrib.BilinearResize2D(x, height=512, width=512)
        return x

if __name__ == '__main__':
    x=mx.nd.ones((16,3,512,512))
    net=ResNet(BottleneckV2,[3, 4, 6, 3],[64, 256, 512, 1024, 2048],16)
    net.initialize()
    print(net(x))





