import torch as pt
import numpy as np
import struct

seed = 2021
np.random.seed(seed)
pt.manual_seed(seed)


def float2binary(num):
    """
    IEEE 754 representation of a 32 bit float
    https://stackoverflow.com/a/16444778
    converts float to a binary
    """
    return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))


def binary2float(b):
    """
    IEEE 754 representation of a 32 bit float
    https://stackoverflow.com/a/42956514
    converts binary b to float
    currently works only for the float values > 0
    """
    return struct.unpack("f", struct.pack("i", int(b, 2)))[0]


def dataset_gen(low=0, high=100, num=100, round_len=5):
    """
    float number dataset generation between two positive ranges
    """
    # the real number
    data = np.random.uniform(
        low, high, num
    )  # random float numbers between 0 and 100
    data = np.round(data, round_len)
    # binary format of the real numbers
    binary = [float2binary(i) for i in data]
    assert data[0] == round(binary2float(binary[0]), 5), "data not matching"
    return data, binary


def prediction(pred, thres=0.65):
    threshold = pt.tensor([thres])
    results = (pred > threshold).float() * 1
    results = results.detach().numpy()
    results = ["".join([str(int(j)) for j in i]) for i in results]
    return results


# # using bitstring module to create float from binary
# val = 78.00234
# import bitstring
# f1 = bitstring.BitArray(float=val, length=32)
# print(f1.bin)


class Model(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pt.nn.Linear(32, 8)
        self.fc2 = pt.nn.Linear(8, 32)
        self.drop = pt.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = pt.nn.functional.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = pt.nn.functional.sigmoid(x)
        return x


batch_size = 32

model = Model().double()
criterion = pt.nn.BCELoss()
opt = pt.optim.Adam(model.parameters(), lr=0.0001)
# opt = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

for i in range(60000):
    opt.zero_grad()
    data, X = dataset_gen(num=batch_size)
    train = pt.from_numpy(np.array([[float(j) for j in list(i)] for i in X]))

    out = model(train)
    loss = criterion(out, train)
    loss.backward()
    opt.step()
    if i % 1000 == 0:
        print("\n Epoch: ", i)
        print("Loss: ", loss.item())
        print()
        with pt.no_grad():
            data, X = dataset_gen(num=3)
            test = pt.from_numpy(
                np.array([[float(j) for j in list(i)] for i in X])
            )
            pred = model(test)
            results = prediction(pred)
            for i, j in zip(data, results):
                try:
                    diff = i - binary2float(j)
                    print(
                        "Real: ", i, " Pred", binary2float(j), " Diff: ", diff
                    )
                except:
                    print("Error")
