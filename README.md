原来模型8卡，batchsize=2，1500的warmup，24000样本输入量
那么我现在4卡，batchsize=2， 3000的warmup，同样24000样本输入量

总epoch，原来是200epoch，每个epoch365个样本，总样本输入量73000个
现在4卡，现在batchsize=2,每个iteration 8个sample，总iteration为9125 iterations