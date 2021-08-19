# 1.简介
本项目基于PaddlePaddle复现《 TabNet: Attentive Interpretable Tabular Learning》论文。通常表格数据都是使用XGBoost和LightGBM这类提升树模型来获得较好的性能。该论文提出了一种使用DNN来处理表格数据，并取得了不错的效果。本文使用PaddlePaddle深度学习框架进行复现，最终在Forest Cover Type数据集达到0.96777的精度，已经超越Pytorch版本的精度。

论文地址：

[https://arxiv.org/pdf/1908.07442v5.pdf](https://arxiv.org/pdf/1908.07442v5.pdf)

参考项目:

[https://github.com/dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)

通过该项目中的[issue](https://github.com/dreamquark-ai/tabnet/issues/70)可知，该项目在Forest Cover Type数据集上的精度为0.9639左右。

# 2.数据集下载

运行程序会自动下载数据集并解压到data目录下，不需要手动下载。

如果想手动下载，地址如下。

Forest Cover Type数据集地址：

[https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz)

# 3.环境

PaddlePaddle == 2.1.2

python == 3.7

还需安装wget自动下载数据集。安装命令如下：
```
pip install wget
```
# 4.训练

1. 训练使用了原文中数据集划分方式，原文参考的论文为《Xgboost: Scalable GPU accelerated learning》。

    相关项目地址：

    [https://github.com/RAMitchell/GBM-Benchmarks/blob/master/benchmark.py](https://github.com/RAMitchell/GBM-Benchmarks/blob/master/benchmark.py)

2. 模型参数保持原文中的参数设置：
    ```
    N_d=N_a=64, λ_sparse=0.0001, B=16384, B_v =512, mB=0.7, N_steps=5 and γ=1.5.
    ```

3. 调整了原文中的训练策略，模型准确率有所提升。使用Warmup+CosineAnnealingDecay方式来调整学习率，最大epoch为3000。每个epoch执行22次迭代。Warmup设置为5000次迭代达到0.02的学习率，CosineAnnealingDecay半周期设置为22 * 3000 - 5000。


4. 训练命令：
    ```
    cd tabnet
    nohup python train.py > tabnet.log &
    ```
    这样程序会后台运行。
    通过以下代码，可以随时查看log。
    ```
    tail -f tabnet.log
    ```

    还可以通过以下命令查看当前保存的最优模型精度。

    ```
    cat tabnet.log | grep -n1 best_model
    ```

    结果如下：
    ```
    --
    67377-epoch 2487| loss: 0.02752 | test_accuracy: 0.9676  |  9:47:54s
    67378:Successfully saved model at output/best_model
    67379-/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
    --
    67648-epoch 2497| loss: 0.02835 | test_accuracy: 0.96773 |  9:50:16s
    67649:Successfully saved model at output/best_model
    67650-/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
    --
    74372-epoch 2746| loss: 0.02596 | test_accuracy: 0.96777 |  10:48:43s
    74373:Successfully saved model at output/best_model
    74374-/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
    ```
    可以从日志中看到 最优模型是在第2746个epoch保存的，当时的loss是0.02596，精度是96.777%

    保存最优的模型代码在paddle_tabnet/callbacks.py文件的133行。

    ```
    def on_epoch_end(self, epoch, logs=None):
    acc = logs['test_accuracy']
    if acc > self.best_acc:
        self.best_weights = copy.deepcopy(self.trainer.network.state_dict())
        self.best_acc = acc
        self.trainer.save_model('output/best_model')
    ```
   

# 5.测试

首先现在最高精度模型文件。(acc: 0.96777)

训练结果模型下载地址：

链接: [https://pan.baidu.com/s/1FdZ1tWEHF7JWTDZqgF1i3Q](https://pan.baidu.com/s/1FdZ1tWEHF7JWTDZqgF1i3Q)

密码: 7hm2


```
unzip best_model.zip
cd tabnet
python predict.py --model_path ../best_model
```

model_path为模型所在的目录地址。

运行结果：

```
FINAL TEST SCORE FOR forest-cover-type : 0.9677719163877009
```


# 6.总结

在本文复现过程遇到了几个问题，虽然都找到了解决办法，但是有的地方还是有些疑惑不知道是不是bug。

1. 在自定义算子中，在预测推理过程中，显存会暴涨，排查结果是因为使用了ctx.save_for_backward(supp_size, output)方法导致的。猜测是在推理过程中只有forward所以保存的Tensor没有被消费掉，所以会暴涨？最终在forward方法中添加一个参数tranning判断是training还是eval，如果是eval阶段则不执行ctx.save_for_backward(supp_size, output)。这样确实内存不会暴涨了，相关[issue](https://github.com/PaddlePaddle/Paddle/issues/34752)。
2. 还是在自定义算子中，使用了grad_input[output == 0] = 0这种语句会导致显存缓缓增加，每次增加的都很少，在迭代一定次数后，显存被占满，最后程序崩溃。通过以下代码代替grad_input[output == 0] = 0最终解决问题，不知道这里是不是bug。
   ```
    idx = paddle.fluid.layers.where(output == 0)
    grad_input_gather = paddle.gather_nd(grad_input, idx)
    grad_input_gather = 0 - grad_input_gather
    grad_input = paddle.scatter_nd_add(grad_input, 
    idx, grad_input_gather)
   ```
3. bn层输入的张量stop_gradient为True时，训练会报错。所以需要处理一下输入才能正常训练，处理方法如下：

    将
    ```
    x = self.initial_bn(x)
    ```
    改为
    ```
    c = paddle.to_tensor(np.array([0]).astype('float32'))
    c.stop_gradient = True
    x_1 = x + c
    x_1.stop_gradient = False
    x = self.initial_bn(x_1)
    ```
以上是遇到的一些问题的总结。