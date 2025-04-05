import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import math
class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = y.reshape([y.shape[0],-1])
        y = self.fc(y)
        y = y.reshape([b,c,1,1])
        out = x * y.expand_as(x)
        return out
class FAE(nn.Layer):
    def __init__(self, bs, Def, stride1, stride2, kernel_size1, kernel_size2, aver_pool, input_channels, output_dim):
        super(FAE, self).__init__()
        self.out_dim = output_dim
        self.bs = bs
        normal_initializer = nn.initializer.Normal(mean=0.0, std=1.0)
        self.A = self.create_parameter(shape=[output_dim],dtype='float32')
        self.k1 = self.create_parameter(
            shape=[input_channels * Def],
            is_bias=False,
            default_initializer=normal_initializer
        )
        self.k2 = self.create_parameter(
            shape=[input_channels * Def],
            is_bias=False,
            default_initializer=normal_initializer
        )
        self.b1_squared = paddle.to_tensor(np.random.rand(), dtype='float32', stop_gradient=True)
        self.b2_squared = paddle.to_tensor(np.random.rand(), dtype='float32', stop_gradient=True)
        self.B1 = self.create_parameter(shape=[input_channels * Def, output_dim], dtype='float32')
        self.B2 = self.create_parameter(shape=[input_channels * Def, output_dim], dtype='float32')
        if aver_pool:
            self.decoder  = nn.Sequential(nn.Conv2DTranspose(in_channels=1, out_channels=input_channels, kernel_size=kernel_size1, stride=stride1, padding=0),
                                          nn.GELU(),
                                          nn.Conv2DTranspose(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size2, stride=stride2, padding=0),
                                          nn.GELU())
        else:
            self.decoder = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2DTranspose(in_channels=1, out_channels=input_channels, kernel_size=kernel_size1, stride=stride1,padding=0),
                nn.GELU(),
                nn.Conv2DTranspose(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel_size2, stride=stride2, padding=0),
                nn.GELU()
            )

    def forward(self, u):
        output1 = F.tanh(self.k1 * u / self.b1_squared)
        output2 = F.tanh(self.k2 * u / self.b2_squared)
        combined = paddle.matmul(output1, self.B1) + paddle.matmul(output2, self.B2)
        final_output = combined + self.A
        final_output = final_output.unsqueeze(1)
        batch_size, uns, outdim =final_output.shape
        final_output = paddle.reshape(final_output,[batch_size,1,int(math.sqrt(self.out_dim)),int(math.sqrt(self.out_dim))])
        outdecoder = self.decoder(final_output)
        return outdecoder, final_output
class FAEWithGrouped(nn.Layer):
    def __init__(self, num_groups, channels_per_group, Def, aver_pool, stride1, stride2, kernel_size1, kernel_size2, feature_size, bat_size=256, reduction=16, output_dim=100, hidden=384, is_cls_token=True, test_sign=True):
        super(FAEWithGrouped, self).__init__()
        self.bs = bat_size
        self.channels_per_group = channels_per_group
        self.num_groups = num_groups
        self.feature_size = feature_size
        self.out_dim = output_dim
        self.Def = Def
        self.aligned_dim = 64
        self.stride1 = stride1
        self.stride2 = stride2
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.is_cls_token = is_cls_token
        self.test_sign = test_sign
        num_tokens = 65
        self.se_layers = nn.LayerList([SELayer(channels_per_group, reduction) for _ in range(num_groups)])
        self.fae_layers = nn.LayerList([FAE(self.bs, self.Def, self.stride1, self.stride2, self.kernel_size1, self.kernel_size2, aver_pool, channels_per_group, output_dim) for _ in range(num_groups)])
        self.emb = nn.Linear(100, hidden)
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, hidden],
            dtype='float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1, 1, hidden]))
        ) if is_cls_token else None
        self.pos_embedding = paddle.create_parameter(
            shape=[1, num_tokens, hidden],
            dtype='float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1, num_tokens, hidden]))
        )
        self.conv1 = nn.Conv2D(num_groups, num_groups, kernel_size=1, bias_attr=False)
        self.conv2 = nn.Conv2D(in_channels=num_groups, out_channels=self.aligned_dim, kernel_size=1, stride=1,padding=0)

    def get_fuzzy_input(self, grouped_features):
        reference_subsets = []
        for idx, feature in enumerate(grouped_features):
            processed_feature = self.se_layers[idx](feature)
            batch_size, channels, height, width = processed_feature.shape
            processed_feature = paddle.reshape(processed_feature, [batch_size*channels, self.feature_size, self.feature_size])
            pixel_variance = paddle.var(processed_feature, axis=0)
            pixel_variance_flat = pixel_variance.flatten()
            max_variance_indices_flat = np.argsort(-pixel_variance_flat)
            reference_pixels_flat = max_variance_indices_flat[:self.Def]
            reference_pixels_2d = np.unravel_index(reference_pixels_flat, pixel_variance.shape)
            reference_subset = paddle.zeros((processed_feature.shape[0], self.Def))
            for d in range(self.Def):
                row = reference_pixels_2d[0][d]
                col = reference_pixels_2d[1][d]
                reference_subset[:, d] = processed_feature[:, row, col]
            reference_subset = paddle.reshape(reference_subset, [batch_size, self.channels_per_group*self.Def])
            reference_subsets.append(reference_subset)
        return reference_subsets

    def forward(self, grouped_features):
        processed_features = []
        output_of_decoders = []
        reference_subsets = self.get_fuzzy_input(grouped_features)
        for idx, feature in enumerate(reference_subsets):
            output_of_decoder, fuzzy_feature = self.fae_layers[idx](feature)
            processed_features.append(fuzzy_feature)
            output_of_decoders.append(output_of_decoder)
        output_of_decoders = paddle.concat(output_of_decoders, axis=1)
        processed_features = paddle.concat(processed_features,axis=1)
        processed_features = self.conv1(processed_features)
        processed_features = self.conv2(processed_features)
        batch_size, channels, height, width = processed_features.shape
        processed_features = paddle.reshape(processed_features, [batch_size, self.aligned_dim, self.out_dim])
        processed_features_emb = self.emb(processed_features)
        if self.is_cls_token:
            processed_features_totrans = paddle.concat([self.cls_token.tile([processed_features_emb.shape[0],1,1]), processed_features_emb], axis=1)
        processed_features_totrans = processed_features_totrans+self.pos_embedding
        return output_of_decoders, processed_features_totrans
class BottleneckBlock2(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock2, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels == out_channels // self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock2_Nods(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock2_Nods, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock2_faeout(nn.Layer):
    expansion = 4
    def __init__(self, bs, test_sign, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock2_faeout, self).__init__()
        self.num_groups = 16
        self.channels_per_groups = int(out_channels / self.num_groups)
        self.bs = bs
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.conv4 = nn.Conv2D(out_channels, out_channels, kernel_size=1, bias_attr=False)
        self.faewithse = FAEWithGrouped(test_sign = test_sign, bat_size=self.bs, num_groups=self.num_groups, channels_per_group=self.channels_per_groups, aver_pool=True, stride1=3, stride2=1, kernel_size1=3, kernel_size2=3, Def=10, feature_size= 32)
        self.relu = nn.ReLU()
        self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out3 = self.bn3(self.conv4(out3))
        num_group = out3.shape[1]//16
        group_features = paddle.chunk(out3, num_group, 1)
        outdecoder, outfae = self.faewithse(group_features)
        out = outdecoder
        out += residual
        out = self.relu(out)
        return out, outfae
class BottleneckBlock3(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock3, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        if stride != 1 or in_channels == out_channels // self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock3_Nods(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock3_Nods, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock3_faeout(nn.Layer):
    expansion = 4
    def __init__(self, bs, test_sign, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock3_faeout, self).__init__()
        self.num_groups = 16
        self.channels_per_groups = int(out_channels / self.num_groups)
        self.bs = bs
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.conv4 = nn.Conv2D(out_channels, out_channels, kernel_size=1, bias_attr=False)
        self.faewithse = FAEWithGrouped(test_sign=test_sign, bat_size=self.bs, num_groups=self.num_groups, channels_per_group=self.channels_per_groups, aver_pool=True, stride1=1, stride2=1, kernel_size1=3, kernel_size2=5, Def=5, feature_size= 16)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out3 = self.bn3(self.conv4(out3))
        num_group = out3.shape[1] // 32
        group_features = paddle.chunk(out3, num_group, 1)
        outdecoder, outfae = self.faewithse(group_features)
        out = outdecoder
        out += residual
        out = self.relu(out)
        return out, outfae
class BottleneckBlock4(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock4, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        if stride != 1 or in_channels == out_channels // self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock4_Nods(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock4_Nods, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock4_faeout(nn.Layer):
    expansion = 4
    def __init__(self, bs, test_sign, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock4_faeout, self).__init__()
        self.num_groups = 16
        self.channels_per_groups = int(out_channels / self.num_groups)
        self.bs = bs
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.conv4 = nn.Conv2D(out_channels, out_channels, kernel_size=1, bias_attr=False)
        self.faewithse = FAEWithGrouped(test_sign=test_sign, bat_size=bs, num_groups=self.num_groups, channels_per_group=self.channels_per_groups, aver_pool=False, stride1=1, stride2=2,kernel_size1=4, kernel_size2=2,Def=2, feature_size= 8)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out3 = self.bn3(self.conv4(out3))
        num_group = out3.shape[1] // 64
        group_features = paddle.chunk(out3, num_group, 1)
        outdecoder, outfae = self.faewithse(group_features)
        out = outdecoder
        out += residual
        out = self.relu(out)
        return out, outfae
class BottleneckBlock5(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock5, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        if stride != 1 or in_channels == out_channels // self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock5_Nods(nn.Layer):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock5_Nods, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out = out3
        out += residual
        out = self.relu(out)
        return out
class BottleneckBlock5_faeout(nn.Layer):
    expansion = 4
    def __init__(self, bs, test_sign, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock5_faeout, self).__init__()
        self.num_groups = 16
        self.channels_per_groups = int(out_channels / self.num_groups)
        self.bs = bs
        self.conv1 = nn.Conv2D(in_channels, out_channels // self.expansion, kernel_size=1, stride=stride,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv2 = nn.Conv2D(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels // self.expansion)
        self.conv3 = nn.Conv2D(out_channels // self.expansion, out_channels, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.conv4 = nn.Conv2D(out_channels, out_channels, kernel_size=1, bias_attr=False)
        self.faewithse = FAEWithGrouped(test_sign=test_sign, bat_size=bs, num_groups=self.num_groups, channels_per_group=self.channels_per_groups, aver_pool=False, stride1=1, stride2=2, kernel_size1=2, kernel_size2=2,Def=1, feature_size= 4)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out3 = self.bn3(self.conv4(out3))
        num_group = out3.shape[1] // 128
        group_features = paddle.chunk(out3, num_group, 1)
        outdecoder, outfae = self.faewithse(group_features)
        out = outdecoder
        out += residual
        out = self.relu(out)
        return out, outfae
def resnet_bn_module2(bs, test_sign, input_channels, out_channels, num_res_blocks, stride=1):
    blk = []
    for i in range(num_res_blocks):
        if i == 0:
            blk.append(BottleneckBlock2(input_channels, out_channels,
                                stride=stride))
        elif i < (num_res_blocks - 1):
            blk.append(BottleneckBlock2_Nods(out_channels, out_channels, downsample=None))
        else:
            blk.append(BottleneckBlock2_faeout(bs, test_sign, out_channels, out_channels, downsample=None))
    return blk
def resnet_bn_module3(bs, test_sign, input_channels, out_channels, num_res_blocks, stride=2):
    blk = []
    for i in range(num_res_blocks):
        if i ==0:
            blk.append(BottleneckBlock3(input_channels, out_channels,
                                stride=stride))
        elif i < (num_res_blocks - 1):
            blk.append(BottleneckBlock3_Nods(out_channels, out_channels, stride=1))
        else:
            blk.append(BottleneckBlock3_faeout(bs,test_sign, out_channels, out_channels, stride=1))
    return blk
def resnet_bn_module4(bs, test_sign, input_channels, out_channels, num_res_blocks, stride=2):
    blk = []
    for i in range(num_res_blocks):
        if i == 0:
            blk.append(BottleneckBlock4(input_channels, out_channels,
                                stride=stride))
        elif i < (num_res_blocks - 1):
            blk.append(BottleneckBlock4_Nods(out_channels, out_channels,stride=1))
        else:
            blk.append(BottleneckBlock4_faeout(bs, test_sign, out_channels, out_channels, stride=1))
    return blk
def resnet_bn_module5(bs, test_sign, input_channels, out_channels, num_res_blocks, stride=2):
    blk = []
    for i in range(num_res_blocks):
        if i == 0:
            blk.append(BottleneckBlock5(input_channels, out_channels,
                                stride=stride))
        elif i < (num_res_blocks - 1):
            blk.append(BottleneckBlock5_Nods(out_channels, out_channels))
        else:
            blk.append(BottleneckBlock5_faeout(bs, test_sign, out_channels, out_channels))
    return blk
def make_first_module(in_channels):
    m1 = nn.Sequential(nn.Conv2D(in_channels, 64, 3, stride=1, padding=1,bias_attr=False),
                    nn.BatchNorm2D(64), nn.ReLU(),)
                    # nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
    return m1
def make_modules(bs, test_sign):
    m2 = nn.Sequential(*resnet_bn_module2( bs, test_sign, 64, 256, 3, stride=1))
    m3 = nn.Sequential(*resnet_bn_module3( bs, test_sign, 256, 512, 4, stride=2))
    m4 = nn.Sequential(*resnet_bn_module4( bs, test_sign, 512, 1024, 6, stride=2))
    m5 = nn.Sequential(*resnet_bn_module5( bs, test_sign, 1024, 2048, 3, stride=2))
    return m2, m3, m4, m5
class Mlp(nn.Layer):
    def __init__(self, feats, mlp_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(feats, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, feats)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, feats, head=8, dropout=0., attn_dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats ** 0.5
        self.qkv = nn.Linear(feats,
                             feats * 3)
        self.out = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.head, self.feats//self.head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)
        attn = F.softmax(paddle.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, axis=-1)
        attn_map = self.attn_dropout(attn)
        attn = paddle.einsum("bhij, bhjf->bihf", attn_map, v)
        out = self.dropout(self.out(attn.flatten(2)))
        return out
class TransformerEncoder(nn.Layer):
    def __init__(self, feats, mlp_hidden, head=8, dropout=0., attn_dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.layer1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, attn_dropout=attn_dropout)
        self.layer2 = nn.LayerNorm(feats)
        self.mlp = Mlp(feats, mlp_hidden)

    def forward(self, x):
        out = self.msa(self.layer1(x)) + x
        out = self.mlp(self.layer2(out)) + out
        return out
class ViT(nn.Layer):
    def __init__(self, in_c=3, dropout=0., attn_dropout=0.0, num_layers=7, hidden=384, mlp_hidden=384*4, head=12):
        super(ViT, self).__init__()
        encoder_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, attn_dropout=attn_dropout, head=head) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*encoder_list)

    def forward(self, x):
        out = self.encoder(x)
        return out
class CrossAttention(nn.Layer):
    def __init__(self, feats, dropout=0.0, attn_dropout=0.0, head=12, is_cls_token=True):
        super(CrossAttention, self).__init__()
        self.is_cls_token = is_cls_token
        self.feats = feats
        self.sqrt_d=feats/head
        self.q=nn.Linear(feats,feats)
        self.kv = nn.Linear(feats, feats * 2)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.head=head
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(feats, feats)

    def transpose_multi_head_q(self, q):
        new_shape = q.shape[:-1] + [self.head, self.feats//self.head]
        mul_q = q.reshape(new_shape)
        mul_q = mul_q.transpose([0, 2, 1, 3])
        return mul_q

    def transpose_multi_head_k(self, k):
        new_shape = k.shape[:-1] + [self.head, self.feats//self.head]
        mul_k = k.reshape(new_shape)
        mul_k = mul_k.transpose([0, 2, 1, 3])
        return mul_k

    def transpose_multi_head_v(self, v):
        new_shape = v.shape[:-1] + [self.head, self.feats//self.head]
        mul_v = v.reshape(new_shape)
        mul_v = mul_v.transpose([0, 2, 1, 3])
        return mul_v

    def forward(self, x):
        q=x[:,0]
        q=q.unsqueeze(1)
        k,v=self.kv(x).chunk(2,-1)
        q=self.transpose_multi_head_q(q)
        k=self.transpose_multi_head_k(k)
        v=self.transpose_multi_head_v(v)
        cross_attn_qk = F.softmax(paddle.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, axis=-1)
        cross_attn_qk = self.attn_dropout(cross_attn_qk)
        attn = paddle.einsum("bhij, bhjf->bihf", cross_attn_qk, v)
        out = self.dropout(self.out(attn.flatten(2)))
        return out
class Mymodel(nn.Layer):
    def __init__(self, bs, in_channels=3, num_classes=10, hidden=384,use_residual=True, is_cls_token=True, test_sign = False ):
        super(Mymodel,self).__init__()
        self.is_cls_token = is_cls_token
        self.feats = hidden
        self.bs = bs
        self.test_sign = test_sign
        num_tokens = 65
        self.emb = nn.Linear(32, hidden)
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, hidden],
            dtype='float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1, 1, hidden]))
        ) if is_cls_token else None
        self.pos_embedding = paddle.create_parameter(
            shape=[1, num_tokens, hidden],
            dtype='float32',
            default_initializer=nn.initializer.Assign(paddle.randn([1, num_tokens, hidden]))
        )
        self.num_classes = num_classes
        self.m1 = make_first_module(in_channels)
        self.m2, self.m3, self.m4, self.m5 = make_modules(bs, test_sign)
        self.adapconv = nn.AdaptiveAvgPool2D(1)
        self.mvit = ViT()
        self.cross_attention = CrossAttention(self.feats)
        self.mlp_head_fcm = nn.Sequential(
            nn.LayerNorm(self.feats),
            nn.Linear(self.feats, num_classes)
        )
        self.mlp_head_mcf = nn.Sequential(
            nn.LayerNorm(self.feats),
            nn.Linear(self.feats, num_classes)
        )
        self.LN = nn.LayerNorm(2048)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out1 = self.m1(x)
        out2, out2_fae = self.m2(out1)
        out3, out3_fae = self.m3(out2)
        out4, out4_fae = self.m4(out3)
        out5, out5_fae= self.m5(out4)
        # out2_fae = self.mvit(out2_fae)
        # out3_fae = self.mvit(out3_fae)
        # out4_fae = self.mvit(out4_fae)
        # out5_fae = self.mvit(out5_fae)
        out_fae_sum_trans = out2_fae+ out3_fae + out4_fae + out5_fae
        out_fae_sum_trans = self.mvit(out_fae_sum_trans)
        out5 = self.adapconv(out5)
        batch_size, channels, h, w =out5.shape
        out5_reshape = paddle.reshape(out5, [batch_size, 64, 32])
        out5_emb = self.emb(out5_reshape)
        if self.is_cls_token:
            out5_trans = paddle.concat([self.cls_token.tile([out5_emb.shape[0], 1, 1]), out5_emb], axis=1)
        out5_trans = out5_trans + self.pos_embedding
        out5_trans_cls_token = out5_trans[:, 0]
        out5_trans_cls_token = out5_trans_cls_token.unsqueeze(1)
        out5_trans_patch_token = out5_trans[:,1:]
        out_fae_sum_trans_cls_token = out_fae_sum_trans[:,0]
        out_fae_sum_trans_cls_token = out_fae_sum_trans_cls_token.unsqueeze(1)
        out_fae_sum_trans_patch_token = out_fae_sum_trans[:,1:]
        out_fcls_cat_patchtoken = paddle.concat([out_fae_sum_trans_cls_token, out5_trans_patch_token], axis=1)
        out_ca_fcls_cat_patchtoken = self.cross_attention(out_fcls_cat_patchtoken)
        out_ca_fcls_cat_patchtoken = out_ca_fcls_cat_patchtoken.squeeze(1)
        out_fcm=self.mlp_head_fcm(out_ca_fcls_cat_patchtoken)
        out_mcls_cat_fpatchtoken = paddle.concat([out5_trans_cls_token, out_fae_sum_trans_patch_token],axis=1)
        out_ca_mcls_cat_fpatchtoken = self.cross_attention(out_mcls_cat_fpatchtoken)
        out_ca_mcls_cat_fpatchtoken = out_ca_mcls_cat_fpatchtoken.squeeze(1)
        out_mcf = self.mlp_head_mcf(out_ca_mcls_cat_fpatchtoken)
        out = out_fcm+out_mcf
        return out
# mymodel = Mymodel(bs=1, in_channels=1, num_classes=10, use_residual=True, test_sign = True)
# FLOPs = paddle.flops(mymodel, (1, 1, 32, 32))
# print(FLOPs)
