# ResNet笔记

[resnet 原论文](<https://arxiv.org/pdf/1512.03385.pdf>)

这篇知乎文章讲的十分详细易懂：<https://zhuanlan.zhihu.com/p/54289848

甚至是只看上面的文章，不用看下面的笔记都行

resenet的输入图像分辨率固定是224*224，同时要求输入channel维度为3

**resnet结构图**

<img src="https://pic1.zhimg.com/80/v2-71633c4b129b187f1c6a198473b30fd8_hd.jpg" width=30%>

（一般服务器显卡只够跑resnet18，无视下图softmax，一般是没有的）

![](https://github.com/zhongzhh8/Picture_markdown/blob/master/resnet.png?raw=true)

![](https://github.com/zhongzhh8/Picture_markdown/blob/master/Snipaste_2019-09-01_16-32-43.png?raw=true)



第一组卷积的输入大小是**224x224**，第五组卷积的输出大小是7x7，每次缩小2倍，总共缩小5次。

**resnet18**和resnet34的输出经过avgpooling后是一个**512**维向量。然后接上一个映射到1000维的fc层。

resnet50、resnet101、resnet152的输出经过avgpooling后是一个**2048**维向量。然后同样接上一个映射到1000维的fc层。

除了使用1x1和7x7卷积之外，其他全部都是用的 3x3卷积。



[pytorch官方的resnet函数地址](<https://pytorch.org/docs/stable/torchvision/models.html?highlight=resnet18#torchvision.models.resnet18>)

![](https://github.com/zhongzhh8/Picture_markdown/blob/master/Snipaste_2019-09-01_16-16-20.png?raw=true)





**resnet官方训练方法**

在ImageNet上的测试设置如下： 
图片使用欠采样放缩到[256∗480][256∗480]，以提供尺寸上的数据增强。对原图作水平翻转，并且使用[224∗224][224∗224]的随机采样，同时每一个像素作去均值处理。在每一个卷积层之后，激活函数之前使用BN。使用SGD，mini-batch大小为256。学习率的初始值为0.1，当训练误差不再缩小时降低学习率为原先的1/10继续训练。训练过程进行了600000次迭代。

**main idea** 

将部分原始输入的信息不经过矩阵乘法和非线性变换，直接传输到下一层。ResNet通过改变学习目标，即不再学习完整的输出F(x)，而是学习残差H(x)−x，解决了传统卷积层或全连接层在进行信息传递时存在的丢失、损耗等问题。通过直接将信息从输入绕道传输到输出，一定程度上保护了信息的完整性。



**shortcut示意图**![](./20180114184946861.png)

**构建恒等映射**（Identity mapping）

简单地说，原先的网络输入x，希望输出H(x)。现在我们改一改，我们令H(x)=F(x)+x，那么我们的网络就只需要学习输出一个残差F(x)=H(x)-x。作者提出，学习残差F(x)=H(x)-x会比直接学习原始特征H(x)简单的多。

**真正的shortcut设计：**

![](https://img-blog.csdn.net/20180114183212429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

resnet18和resnet34用的是左边的常规设计，resnet50及以上的用右边的设计，它先用1x1卷积将feature map的channel数从256降到64 ，经过3x3卷积后，再用1x1卷积将channel数恢复回256。

整个shortcut结构叫building block，右边这种技巧叫bottleneck design





