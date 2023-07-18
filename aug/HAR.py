import math
import tensorflow as tf
import numpy as np
import random
import scipy
import torch
from sklearn.utils import check_random_state


def resample(x, M, N):
    from scipy.interpolate import interp1d
    timesetps = x.shape[1]
    orig_steps = np.arange(timesetps)
    interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / (M + 1))
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)

    length_inserted = InterpVal.shape[1]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    return InterpVal[:, index_selected, :]


def resample_random(x):
    M, N = random.choice([[1, 0], [2, 1], [3, 2]])
    from scipy.interpolate import interp1d
    timesetps = x.shape[1]
    orig_steps = np.arange(timesetps)
    interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / (M + 1))
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)

    length_inserted = InterpVal.shape[1]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    return InterpVal[:, index_selected, :]


def noise(x):
    x = tf.add(x, tf.multiply(x, tf.cast(
        tf.random.uniform(shape=(x.shape[0], x.shape[1], x.shape[2]), minval=-0.1, maxval=0.1), tf.float64)))
    return x


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def rotate(x, angles=np.pi / 12):
    t = angles
    f = angles
    r = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(t), -np.sin(t)],
                   [0, np.sin(t), np.cos(t)]])
    Ry = np.array([[np.cos(f), 0, np.sin(f)],
                   [0, 1, 0],
                   [-np.sin(f), 1, np.cos(f)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    c = x.shape[2] // 3
    x_new = np.matmul(np.matmul(np.matmul(Rx, Ry), Rz), np.transpose(x[:, :, 0:3], (0, 2, 1))).transpose((0, 2, 1))
    for i in range(1, c):
        temp = np.matmul(np.matmul(np.matmul(Rx, Ry), Rz),
                         np.transpose(x[:, :, i * 3:i * 3 + 3], (0, 2, 1))).transpose((0, 2, 1))
        x_new = np.concatenate((x_new, temp), axis=-1)
    return x_new


def scaling(x):
    alpha = np.random.randint(7, 10) / 10
    # alpha = 0.9
    return tf.multiply(x, alpha)


#
def magnify(x):
    lam = np.random.randint(11, 14) / 10
    return tf.multiply(x, lam)


def get_cubic_spline_interpolation(x):
    cubic_spline = scipy.interpolate.CubicSpline(np.arange(0, x.shape[1]), x[0, :, 0])
    return cubic_spline(np.arange(0.5, x.shape[1] - 1))


def inverting(x):
    return np.multiply(x, -1)


def reversing(x):
    return x[:, -1::-1, :]


def rotation(x):
    c = x.shape[2] // 3
    x_new = rotation_transform_vectorized(x[:, :, 0:3])
    for i in range(1, c):
        temp = rotation_transform_vectorized(x[:, :, i * 3:(i + 1) * 3])
        x_new = np.concatenate((x_new, temp), axis=-1)
    return x_new


def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

    return np.matmul(X, matrices)


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    m = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])
    matrix_transposed = np.transpose(m, axes=(2, 0, 1))
    return matrix_transposed


from scipy.interpolate import lagrange, CubicSpline


# 单条曲线插值
def single_lagrange(x):
    # x is a one axis
    # 以四个点拟合一条曲线
    # 取第2.5个值
    q = 4
    k = 1.5
    l = len(x)
    new_v = []
    x_axis = np.arange(q)
    for i in range(l // q):
        new_v.append(lagrange(x_axis, x[i * q:(i + 1) * q])(k))
    insert_index = np.arange(2, l, 4)
    x = np.insert(x, insert_index, new_v)
    return x


def multi_lagrange(x):
    # x is a one axis
    # 以8个点拟合一条曲线
    # 取第1.5,3.5,5.5个值
    q = 8
    k = [1.5, 3.5, 5.5]
    ind = [2, 4, 6]
    l = len(x)
    new_v = []
    x_axis = np.arange(q)
    insert_index = []
    for i in range(l // q):
        lar = lagrange(x_axis, x[i * q:(i + 1) * q])
        for j in k:
            new_v.append(lar(j))
        for l in ind:
            insert_index.append(i * q + l)
    # insert_index = np.arange(2, l, 4)
    x = np.insert(x, insert_index, new_v)
    return x


# 拉格朗自插值法
def mylagrange(x, func):
    # 增采样
    length = x.shape[1]
    channels = x.shape[-1]
    batch_insert = []
    for i in x:
        channel_insert = []
        for j in range(channels):
            new_channel = func(i[:, j])
            channel_insert.append(new_channel)
        batch_insert.append(channel_insert)

    batch_insert = np.array(batch_insert).transpose((0, 2, 1))

    # 减采样
    start = random.randint(0, batch_insert.shape[1] - length - 1)
    index = np.arange(start, start + length, 1)
    return batch_insert[:, index, :]


def single_Cubic(x):
    # x is a one axis
    # 以四个点拟合一条曲线
    # 取第2.5个值
    q = 4
    k = 1.5
    l = len(x)
    new_v = []
    x_axis = np.arange(q)
    for i in range(l // q):
        new_v.append(CubicSpline(x_axis, x[i * q:(i + 1) * q])(k))
    insert_index = np.arange(2, l, 4)
    x = np.insert(x, insert_index, new_v)
    return x


def multi_Cubic(x):
    # x is a one axis
    # 以8个点拟合一条曲线
    # 取第1.5,3.5,5.5个值
    q = 8
    k = [1.5, 3.5, 5.5]
    ind = [2, 4, 6]
    l = len(x)
    new_v = []
    x_axis = np.arange(q)
    insert_index = []
    for i in range(l // q):
        cub = CubicSpline(x_axis, x[i * q:(i + 1) * q])
        for j in k:
            new_v.append(cub(j))
        for l in ind:
            insert_index.append(i * q + l)
    # insert_index = np.arange(2, l, 4)
    x = np.insert(x, insert_index, new_v)
    return x


# 三次样条
def myCubic(x, func):
    # 增采样
    length = x.shape[1]
    channels = x.shape[-1]
    batch_insert = []
    for i in x:
        channel_insert = []
        for j in range(channels):
            # new_channel = multi_Cubic(i[:,j])
            new_channel = func(i[:, j])
            channel_insert.append(new_channel)
        batch_insert.append(channel_insert)

    batch_insert = np.array(batch_insert).transpose((0, 2, 1))
    # 减采样
    start = random.randint(0, batch_insert.shape[1] - length - 1)
    index = np.arange(start, start + length, 1)
    return batch_insert[:, index, :]


def resampling(x, M, N):
    """
    :param x: the data of a batch,shape=(batch_size,timesteps,features)
    :param M: the number of  new value under tow values
    :param N: the interval of resampling
    :return: x after resampling，shape=(batch_size,timesteps,features)
    """
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]

    for i in range(timesetps - 1):
        x1 = x[:, i * (M + 1), :]
        x2 = x[:, i * (M + 1) + 1, :]
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            x = np.insert(x, i * (M + 1) + j + 1, v, axis=1)

    length_inserted = x.shape[1]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    return x[:, index_selected, :]


def resampling_random(x):
    import random
    M = random.randint(1, 3)
    N = random.randint(0, M - 1)
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]

    for i in range(timesetps - 1):
        x1 = x[:, i * (M + 1), :]
        x2 = x[:, i * (M + 1) + 1, :]
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            x = np.insert(x, i * (M + 1) + j + 1, v, axis=1)
    length_inserted = x.shape[1]
    num = x.shape[0]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    x_selected = x[0, index_selected, :][np.newaxis,]
    for k in range(1, num):
        start = random.randint(0, length_inserted - timesetps * (N + 1))
        index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
        x_selected = np.concatenate((x_selected, x[k, index_selected, :][np.newaxis,]), axis=0)
    return x_selected


def fast_resampling(x, M=1, N=0):
    '''
        :param x: the data of a batch,shape=(batch_size,timesteps,features)
        :param M: the number of  new value under tow values
        :param N: the interval of resampling
        :return: x after resampling，shape=(batch_size,timesteps,features)
        '''
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]
    start = random.randint(0, timesetps - timesetps // (M + 1) - 1)
    end = timesetps // (M + 1) + start
    for i in range(start, end):
        x1 = x[:, start + (i - start) * (M + 1), :]
        x2 = x[:, start + (i - start) * (M + 1) + 1, :]
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            x = np.insert(x, start + (i - start) * (M + 1) + j + 1, v, axis=1)

    return x[:, start:start + timesetps, :]


# 遮掩一个传感器数据
# 选几个通道加噪音 ?
class MaskSense(object):
    def __init__(self, sense=6, num=1, seed=10):
        if num == 'auto' or num >= sense:
            num = np.random.randint(0, sense, size=1, dtype=np.int32)
        # random_seed = check_random_state(seed)
        # 这里使用choice，选择的通道依然具有很大的随机性，因此我打算试试确定的随机种子，减免这个的影响
        self.mask = np.random.choice(range(sense), num, replace=False)
        # self.mask = random_seed.choice(range(sense), num, replace=False)

    def __call__(self, x):
        assert len(x.shape) == 3
        for i in self.mask:
            x[:, :, i] = 0
        return x


def mask_sense(x, sense_list):
    x[:, :, sense_list] = 0.0
    return x


def mask_features(x, features_list):
    x[:, features_list, :] = 0.0
    return x


def sense_noise(x, sense_list, sigma=0.8):
    y = np.random.normal(loc=0., scale=sigma, size=x[:, :, sense_list].shape).astype(np.float32)
    x[:, :, sense_list] += y
    return x


def features_noise(x, features_list, sigma=0.8):
    y = np.random.normal(loc=0., scale=sigma, size=x[:, features_list, :].shape).astype(np.float32)
    x[:, features_list, :] += y
    return x


def random_mask_sense(x, num=1, seed=10, type='drop'):
    # drop noise
    import numpy as np
    sense_num = x.shape[-1]
    # random_seed = check_random_state(seed)
    # 这里使用choice，选择的通道依然具有很大的随机性，因此我打算试试确定的随机种子，减免这个的影响
    sense_list = np.random.choice(sense_num, num, replace=False)
    # sense_list = random_seed.choice(sense_num, num, replace=False)
    if type == 'drop':
        return mask_sense(x, sense_list)
    elif type == 'noise':
        return sense_noise(x, sense_list)
    else:
        return x


def random_mask_feature(x, num, seed=10, type='drop'):
    import numpy as np
    features_num = x.shape[1]
    # num = int(features_num * num)
    # random_seed = check_random_state(seed)
    # 这里使用choice，选择的通道依然具有很大的随机性，因此我打算试试确定的随机种子，减免这个的影响
    # features_list = random_seed.choice(features_num, num, replace=False)
    features_list = np.random.choice(features_num, num, replace=False)
    if type == 'drop':
        return mask_features(x, features_list)
    elif type == 'noise':
        return features_noise(x, features_list)
    else:
        return x


# HAR
class Resample(object):
    def __init__(self, M, N):
        self.M = M
        self.N = N

    def __call__(self, sample):
        from scipy.interpolate import interp1d
        timesetps = sample.shape[1]
        orig_steps = np.arange(timesetps)

        interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / (self.M + 1))

        Interp = interp1d(orig_steps, sample, axis=1)

        InterpVal = Interp(interp_steps)

        length_inserted = InterpVal.shape[1]

        start = random.randint(0, length_inserted - timesetps * (self.N + 1))

        index_selected = np.arange(
            start, start + timesetps * (self.N + 1), self.N + 1)

        if len(InterpVal.shape) == 2:
            return torch.from_numpy(InterpVal[:, index_selected]).float()
        else:
            return torch.from_numpy(InterpVal[:, index_selected, :]).float()


class ResampleRandom(object):
    def __init__(self):
        super(ResampleRandom).__init__()

    def __call__(self, sample):
        from scipy.interpolate import interp1d
        m, n = random.choice([[1, 0], [2, 1], [3, 2]])
        if type(sample) == list:
            sample = np.array(sample)
        timesetps = sample.shape[1]
        orig_steps = np.arange(timesetps)
        interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / (m + 1))
        Interp = interp1d(orig_steps, sample, axis=1)
        InterpVal = Interp(interp_steps)
        length_inserted = InterpVal.shape[1]
        start = random.randint(0, length_inserted - timesetps * (n + 1))
        index_selected = np.arange(
            start, start + timesetps * (n + 1), n + 1)
        if len(InterpVal.shape) == 2:
            return torch.from_numpy(InterpVal[:, index_selected]).float()
        else:
            return torch.from_numpy(InterpVal[:, index_selected, :]).float()


def insert_noise(x, seed=10, alpha=0.1):
    time_steps = x.shape[1]
    features = x.shape[2]
    noise_ts = math.floor(time_steps * alpha)
    noise_features = math.floor(features * alpha)
    noise_t = np.random.normal(loc=0., scale=1., size=(noise_ts, features))
    noise_f = np.random.normal(
        loc=0., scale=1., size=(time_steps, noise_features))

    # random_seed = check_random_state(seed)
    # 这里使用choice，选择的通道依然具有很大的随机性，因此我打算试试确定的随机种子，减免这个的影响
    inx = np.random.choice(np.arange(0, time_steps - noise_ts), size=1)
    # inx = random_seed.choice(np.arange(0, time_steps - noise_ts), size=1)
    new_x = np.insert(x, inx, noise_t, axis=1)
    return new_x[:, :time_steps, :]
