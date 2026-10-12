import torch
import torch.nn as nn
from transformers import AutoModel
import pennylane as qml
import torch.nn.functional as F
import numpy as np
from torchvision.models import ResNet50_Weights, resnet50


class Temp1(nn.Module):
  def __init__(self, args):
    super(Temp1, self).__init__()

    self.qubits = args.qubits
    self.qlayers = args.qlayers
    self.args = args
    self.data_name = args.data_name
    self.enc = args.enc

    if args.enc == "amp":
      if args.data_name == "ahe":
        self.fc_i_res = nn.Linear(2048 * 49, 2 ** args.qubits)
      elif args.data_name == "img":
        self.fc_i_res = nn.Linear(512 * 49, 2 ** args.qubits)

    self.qnn = self.ql()

    if args.data_name == "ahe":
      self.lm1 = nn.LayerNorm(2048 * 49)
    elif args.data_name == "img":
      self.lm1 = nn.LayerNorm(512 * 49)

    self.lm2 = nn.LayerNorm(self.qubits)

    self.fc_q = nn.Linear(self.qubits, 5)
    self.name = "temp1-q=" + str(self.qubits) + "-layers=" + str(self.qlayers) + "-enc=" + str(args.enc)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x_img):
    if self.data_name == "ahe":
      x_img = x_img.view(-1, 49, 2048)
      x_img_res = x_img.reshape(x_img.shape[0], 2048 * 49)
    elif self.data_name == "img":
      x_img = x_img.view(-1, 49, 512)
      x_img_res = x_img.reshape(x_img.shape[0], 512 * 49)

    x_img_res = self.lm1(x_img_res)

    if self.enc == "amp":
      x_img_res = self.fc_i_res(x_img_res)

    x_q = self.qnn(x_img_res)
    x_q = self.lm2(x_q)

    x_q = self.fc_q(x_q)

    return F.log_softmax(x_q, -1)


  def ql(self):
    dev = qml.device("default.qubit", wires=self.qubits)
    @qml.qnode(dev)
    def circuits(inputs, weights):
      if self.args.enc == "amp":
        qml.AmplitudeEmbedding(inputs, wires=range(self.qubits), normalize=True)

      for layer in range(self.qlayers):
        for wire in range(self.qubits):
          qml.RY(weights[layer][wire], wires=wire)

        for i in range(self.qubits)[::2]:
          qml.CNOT(wires=[i, i + 1])

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][self.qubits + idx], wires=wire)

        for i in range(self.qubits - 2)[::2]:
          qml.CNOT(wires=[i + 1, i + 2])

      res = [qml.expval(qml.PauliZ(wires=i)) for i in range(self.qubits)]
      return res

    weight_shapes = {"weights": (self.qlayers, 2 * self.qubits - 2)}
    qnn = qml.qnn.TorchLayer(
      circuits,
      weight_shapes,
      init_method=torch.nn.init.normal_,
    )

    return qnn


class Temp2(nn.Module):
  def __init__(self, args):
    super(Temp2, self).__init__()

    self.qubits = args.qubits
    self.qlayers = args.qlayers
    self.args = args
    self.data_name = args.data_name
    self.enc = args.enc

    if args.enc == "amp":
      if args.data_name == "ahe":
        self.fc_i_res = nn.Linear(2048 * 49, 2 ** args.qubits)
      elif args.data_name == "img":
        self.fc_i_res = nn.Linear(512 * 49, 2 ** args.qubits)

    self.qnn = self.ql()

    if args.data_name == "ahe":
      self.lm1 = nn.LayerNorm(2048 * 49)
    elif args.data_name == "img":
      self.lm1 = nn.LayerNorm(512 * 49)

    self.lm2 = nn.LayerNorm(self.qubits)

    self.fc_q = nn.Linear(self.qubits, 5)
    self.name = "temp2-q=" + str(self.qubits) + "-layers=" + str(self.qlayers) + "-enc=" + str(args.enc)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x_img):
    if self.data_name == "ahe":
      x_img = x_img.view(-1, 49, 2048)
      x_img_res = x_img.reshape(x_img.shape[0], 2048 * 49)
    elif self.data_name == "img":
      x_img = x_img.view(-1, 49, 512)
      x_img_res = x_img.reshape(x_img.shape[0], 512 * 49)

    x_img_res = self.lm1(x_img_res)

    if self.enc == "amp":
      x_img_res = self.fc_i_res(x_img_res)


    x_q = self.qnn(x_img_res)
    x_q = self.lm2(x_q)
    x_q = self.fc_q(x_q)


    return F.log_softmax(x_q, -1)


  def ql(self):
    dev = qml.device("default.qubit", wires=self.qubits)
    @qml.qnode(dev)
    def circuits(inputs, weights):
      if self.args.enc == "amp":
        qml.AmplitudeEmbedding(inputs, wires=range(self.qubits), normalize=True)

      for layer in range(self.qlayers):
        for wire in range(self.qubits):
          qml.RY(weights[layer][wire], wires=wire)

        for i in range(self.qubits)[::2]:
          qml.CNOT(wires=[i + 1, i])

        for wire in range(self.qubits):
          qml.RY(weights[layer][self.qubits + wire], wires=wire)

        for i in range(self.qubits)[::2]:
          qml.CNOT(wires=[i, i + 1])

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][2 * self.qubits + idx], wires=wire)

        for i in range(self.qubits - 2)[::2]:
          qml.CNOT(wires=[i + 2, i + 1])

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][3 * self.qubits - 2 + idx], wires=wire)

        for i in range(self.qubits - 2)[::2]:
          qml.CNOT(wires=[i + 1, i + 2])

      # 进行量子测量；
      res = [qml.expval(qml.PauliZ(wires=i)) for i in range(self.qubits)]
      return res

    weight_shapes = {"weights": (self.qlayers, 4 * self.qubits - 4)}
    qnn = qml.qnn.TorchLayer(
      circuits,
      weight_shapes,
      init_method=torch.nn.init.normal_,
    )

    return qnn


class Temp3(nn.Module):
  def __init__(self, args):
    super(Temp3, self).__init__()

    self.qubits = args.qubits
    self.qlayers = args.qlayers
    self.args = args
    self.data_name = args.data_name
    self.enc = args.enc

    if args.enc == "amp":
      if args.data_name == "ahe":
        self.fc_i_res = nn.Linear(2048 * 49, 2 ** args.qubits)
      elif args.data_name == "img":
        self.fc_i_res = nn.Linear(512 * 49, 2 ** args.qubits)

    self.qnn = self.ql()

    if args.data_name == "ahe":
      self.lm1 = nn.LayerNorm(2048 * 49)
    elif args.data_name == "img":
      self.lm1 = nn.LayerNorm(512 * 49)

    self.lm2 = nn.LayerNorm(self.qubits)

    self.fc_q = nn.Linear(self.qubits, 5)
    self.name = "temp3-q=" + str(self.qubits) + "-layers=" + str(self.qlayers) + "-enc=" + str(args.enc)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x_img):
    if self.data_name == "ahe":
      x_img = x_img.view(-1, 49, 2048)
      x_img_res = x_img.reshape(x_img.shape[0], 2048 * 49)
    elif self.data_name == "img":
      x_img = x_img.view(-1, 49, 512)
      x_img_res = x_img.reshape(x_img.shape[0], 512 * 49)

    x_img_res = self.lm1(x_img_res)

    if self.enc == "amp":
      x_img_res = self.fc_i_res(x_img_res)


    x_q = self.qnn(x_img_res)
    x_q = self.lm2(x_q)
    x_q = self.fc_q(x_q)


    return F.log_softmax(x_q, -1)


  def ql(self):
    dev = qml.device("default.qubit", wires=self.qubits)
    @qml.qnode(dev)
    def circuits(inputs, weights):
      if self.args.enc == "amp":
        qml.AmplitudeEmbedding(inputs, wires=range(self.qubits), normalize=True)

      for layer in range(self.qlayers):
        offset = 0

        for wire in range(self.qubits):
          qml.RY(weights[layer][offset + wire], wires=wire)
        offset += self.qubits

        for idx, wire in enumerate(range(self.qubits)[::2]):
          qml.CRZ(weights[layer][offset + idx], wires=[wire + 1, wire])
        offset += self.qubits // 2

        for wire in range(self.qubits):
          qml.RY(weights[layer][offset + wire], wires=wire)
        offset += self.qubits

        for idx, wire in enumerate(range(self.qubits)[::2]):
          qml.CRZ(weights[layer][offset + idx], wires=[wire, wire + 1])
        offset += self.qubits // 2

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][offset + idx], wires=wire)
        offset += self.qubits - 2

        for idx, wire in enumerate(range(self.qubits - 2)[::2]):
          qml.CRZ(weights[layer][offset + idx], wires=[wire + 2, wire + 1])
        offset += (self.qubits - 2) // 2

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][offset + idx], wires=wire)
        offset += self.qubits - 2

        for idx, wire in enumerate(range(self.qubits - 2)[::2]):
          qml.CRZ(weights[layer][offset + idx], wires=[wire + 1, wire + 2])

      res = [qml.expval(qml.PauliZ(wires=i)) for i in range(self.qubits)]
      return res

    weight_shapes = {"weights": (self.qlayers, 6 * self.qubits - 6)}
    qnn = qml.qnn.TorchLayer(
      circuits,
      weight_shapes,
      init_method=torch.nn.init.normal_,
    )

    return qnn


class Temp4(nn.Module):
  def __init__(self, args):
    super(Temp4, self).__init__()

    self.qubits = args.qubits
    self.qlayers = args.qlayers
    self.args = args
    self.data_name = args.data_name
    self.enc = args.enc

    if args.enc == "amp":
      if args.data_name == "ahe":
        self.fc_i_res = nn.Linear(2048 * 49, 2 ** args.qubits)
      elif args.data_name == "img":
        self.fc_i_res = nn.Linear(512 * 49, 2 ** args.qubits)

    self.qnn = self.ql()

    if args.data_name == "ahe":
      self.lm1 = nn.LayerNorm(2048 * 49)
    elif args.data_name == "img":
      self.lm1 = nn.LayerNorm(512 * 49)

    self.lm2 = nn.LayerNorm(self.qubits)

    self.fc_q = nn.Linear(self.qubits, 5)
    self.name = "temp4-q=" + str(self.qubits) + "-layers=" + str(self.qlayers) + "-enc=" + str(args.enc)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x_img):
    if self.data_name == "ahe":
      x_img = x_img.view(-1, 49, 2048)
      x_img_res = x_img.reshape(x_img.shape[0], 2048 * 49)
    elif self.data_name == "img":
      x_img = x_img.view(-1, 49, 512)
      x_img_res = x_img.reshape(x_img.shape[0], 512 * 49)

    x_img_res = self.lm1(x_img_res)

    if self.enc == "amp":
      x_img_res = self.fc_i_res(x_img_res)

    x_q = self.qnn(x_img_res)
    x_q = self.fc_q(x_q)


    return F.log_softmax(x_q, -1)


  def ql(self):
    dev = qml.device("default.qubit", wires=self.qubits)
    @qml.qnode(dev)
    def circuits(inputs, weights):
      if self.args.enc == "amp":
        qml.AmplitudeEmbedding(inputs, wires=range(self.qubits), normalize=True)


      for layer in range(self.qlayers):
        offset = 0

        for wire in range(self.qubits):
          qml.RY(weights[layer][offset + wire], wires=wire)
        offset += self.qubits

        for idx, wire in enumerate(range(self.qubits)[::2]):
          qml.CNOT(wires=[wire, wire+1])

        for wire in range(self.qubits):
          qml.RY(weights[layer][offset + wire], wires=wire)
        offset += self.qubits

        for idx, wire in enumerate(range(self.qubits)[::2]):
          qml.CNOT(wires=[wire, wire + 1])

        for wire in range(self.qubits):
          qml.RY(weights[layer][offset + wire], wires=wire)
        offset += self.qubits

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][offset + idx], wires=wire)
        offset += self.qubits - 2

        for idx, wire in enumerate(range(self.qubits - 2)[::2]):
          qml.CNOT(wires=[wire + 1, wire + 2])

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][offset + idx], wires=wire)
        offset += self.qubits - 2

        for idx, wire in enumerate(range(self.qubits - 2)[::2]):
          qml.CNOT(wires=[wire + 1, wire + 2])

        for idx, wire in enumerate(range(1, self.qubits - 1)):
          qml.RY(weights[layer][offset + idx], wires=wire)
        offset += self.qubits - 2

      res = [qml.expval(qml.PauliZ(wires=i)) for i in range(self.qubits)]
      return res

    weight_shapes = {"weights": (self.qlayers, 6 * self.qubits - 6)}
    qnn = qml.qnn.TorchLayer(
      circuits,
      weight_shapes,
      init_method=torch.nn.init.normal_,
    )

    return qnn
