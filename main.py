import os
import json
import torch
import numpy as np
from typing_extensions import runtime
from unicodedata import digit
import pickle
from model import *
from data_loader import ReDataset, ReCollate
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim import AdamW
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import *


class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 dev_loader_=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None,
                 ratio=1,
                 enc="amp",
                 train_time=None,
                 seed_num=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.dev_loader_ = dev_loader_
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs
        self.ratio = ratio
        self.enc = enc
        self.train_time = train_time
        self.seed_num = seed_num

    def train(self):
        global_step = 1
        f1_best = 0.0
        f1_dev = []
        runtimes = []
        train_loss, dev_loss = [], []

        for epoch in range(1, self.epochs + 1):
            train_loss_per_lis = []
            s_time = time.time()
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                inputs_i, targets = batch_data[0], batch_data[1]
                inputs_i, targets = inputs_i.to(self.device), targets.to(self.device)
                outputs = self.model(inputs_i)
                loss = F.nll_loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()}")
                global_step += 1
                train_loss_per_lis.append(loss.item())
            train_loss_per = sum(train_loss_per_lis) / len(train_loss_per_lis)
            train_loss.append(train_loss_per)

            print(f"============train_loss: {train_loss_per:.3f}=================")
            e_time = time.time()
            runtime_single = e_time - s_time

            runtimes.append(runtime_single)
            print(f"============Runtime: {runtime_single:.3f} s=================")
            f1 = self.dev()
            f1_dev.append(f1)
            print(f1)
            if f1 > f1_best:
                f1_best = f1
                print(f"******f1_best: {f1_best}******")

                p = os.path.join(self.output_dir, self.model.name + "-" + self.enc + "/")
                if not os.path.exists(p):
                    os.makedirs(p)
                torch.save(self.model.state_dict(),
                           os.path.join(p, "pytorch_model_time_{}_seed_{}.bin".format(self.train_time, self.seed_num)))
                print("****** SAVE ******")

        print(f">>>>>>>>>>>>AVG Runtime: {np.mean(runtimes):.3f} s<<<<<<<<<<<<<<<<")

        return f1_dev, train_loss, np.mean(runtimes)


    def dev(self):
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.dev_loader)):
            inputs_i, targets = batch_data[0], batch_data[1]
            inputs_i, targets = inputs_i.to(self.device), targets.to(self.device)
            outputs = self.model(inputs_i)

            preds.append(outputs)
            trues.append(targets)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        report = classification_report(trues.cpu(), preds.cpu().argmax(axis=-1), digits=4, output_dict=True)
        report_ = classification_report(trues.cpu(), preds.cpu().argmax(axis=-1), digits=4)
        print(report_)
        f1 = report["macro avg"]["f1-score"]
        return f1


    def test(self):
        p = os.path.join(self.output_dir, self.model.name + "-" + self.enc + "/")
        self.model.load_state_dict(torch.load(os.path.join(p, "pytorch_model_time_{}_seed_{}.bin".format(self.train_time, self.seed_num))))
        self.model.eval()
        preds = []
        trues = []

        for step, batch_data in enumerate(tqdm(self.test_loader)):
            inputs_i, targets = batch_data[0], batch_data[1]
            inputs_i, targets = inputs_i.to(self.device), targets.to(self.device)
            outputs = self.model(inputs_i)

            preds.append(outputs)
            trues.append(targets)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        report_dic = classification_report(trues.cpu(), preds.cpu().argmax(axis=-1), digits=4, output_dict=True)
        report = classification_report(trues.cpu(), preds.cpu().argmax(axis=-1), digits=4)

        return report, report_dic, trues, preds


class ClassicalTrainer:
    def __init__(self, model, train_data, val_data, test_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    @staticmethod
    def _report(labels, predictions):
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        report_dict = classification_report(
            labels, predictions, digits=4, output_dict=True, zero_division=0
        )
        report_text = classification_report(labels, predictions, digits=4, zero_division=0)
        return report_text, report_dict

    def train(self):
        start_time = time.time()
        self.model.fit(self.train_data.imgs, self.train_data.labels)
        runtime = time.time() - start_time

        _, validation_report = self._report(
            self.val_data.labels, self.model.predict(self.val_data.imgs)
        )
        validation_f1 = validation_report["macro avg"]["f1-score"]
        print(f"============Fit runtime: {runtime:.3f} s=================")
        print(f"============Validation macro-F1: {validation_f1:.4f}=================")
        return [validation_f1], [], runtime

    def test(self):
        predictions = self.model.predict(self.test_data.imgs)
        report, report_dict = self._report(self.test_data.labels, predictions)
        return report, report_dict, np.asarray(self.test_data.labels), predictions


def build_optimizer_and_scheduler(args, model):
    module = (
        model.module if hasattr(model, "module") else model
    )


    if hasattr(module, "optimizer_parameter_groups"):
        optimizer_parameters = module.optimizer_parameter_groups()
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.Adam(optimizer_parameters, lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


    return optimizer, scheduler


def read_pickle(path):
    with open(path, 'rb')as f:
        feats = pickle.load(f)
    return feats

def save_pickle(path, feats):
    with open(path, 'wb')as f:
        pickle.dump(feats, f)


def main(data_name, t, seed_num, args):
    print("layer", args.qlayers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_name = data_name

    model = Temp1(args)

    print(f"****** {model.name} ******")

    if data_name == "ahe":
        if model.name == "vgg":
            imgs_labs = read_pickle("pkl/ahe_data_vgg_dic.pickle")
        elif model.name in ("resnet", "resnet_new"):
            imgs_labs = read_pickle("pkl/ahe_data_resnet_dic.pickle")
        else:
            imgs_labs = read_pickle("pkl/ahe_data_xception_dic.pickle")
        trainX_img, trainY = imgs_labs["train"]["img"], imgs_labs["train"]["labs"]
        valX_img, valY = imgs_labs["val"]["img"], imgs_labs["val"]["labs"]
        testX_img, testY = imgs_labs["test1"]["img"], imgs_labs["test1"]["labs"]

    elif data_name == "img":
        if model.name == "xception":
            imgs_labs = read_pickle("pkl/img_data_xception_dic.pickle")
        elif model.name in ("resnet", "resnet_new"):
            imgs_labs = read_pickle("pkl/img_data_resnet_dic.pickle")
        else:
            imgs_labs = read_pickle("pkl/img_data_vgg_dic.pickle")
        trainX_img, trainY = imgs_labs["train"]["img"], imgs_labs["train"]["labs"]
        valX_img, valY = imgs_labs["val"]["img"], imgs_labs["val"]["labs"]
        testX_img, testY = imgs_labs["test1"]["img"], imgs_labs["test1"]["labs"]


    trainData = LpDataset(imgs=trainX_img, labels=trainY)
    valData = LpDataset(imgs=valX_img, labels=valY)
    testData = LpDataset(imgs=testX_img, labels=testY)

    trainLoader = DataLoader(dataset=trainData, batch_size=args.train_batch_size,
                             sampler=torch.utils.data.RandomSampler(trainData),
                             num_workers=0,
                             pin_memory=True)

    valLoader = DataLoader(dataset=valData, batch_size=args.train_batch_size,
                             sampler=torch.utils.data.RandomSampler(valData),
                             num_workers=0,
                             pin_memory=True)

    testLoader = DataLoader(dataset=testData, batch_size=args.train_batch_size,
                            sampler=torch.utils.data.RandomSampler(testData),
                            num_workers=0,
                            pin_memory=True)



    if getattr(model, "is_classical", False):
        train = ClassicalTrainer(
            model=model,
            train_data=trainData,
            val_data=valData,
            test_data=testData,
        )
    else:
        model.to(device)
        optimizer, schedule = build_optimizer_and_scheduler(args, model)
        train = Trainer(
            output_dir=args.output_dir,
            model=model,
            train_loader=trainLoader,
            dev_loader=valLoader,
            test_loader=testLoader,
            optimizer=optimizer,
            schedule=schedule,
            epochs=args.epochs,
            device=device,
            ratio=args.ratio,
            enc=args.enc,
            train_time=t,
            seed_num=seed_num,
        )

    f1_dev, train_loss, running_time = train.train()
    report, report_dic, trues, preds = train.test()
    print("====== Performance in Testing set ====== \n")
    print(report)

    return report_dic, f1_dev, model.name, train_loss, trues, preds, running_time


if __name__ == "__main__":

    data_name = "ahe"
    print(data_name)

    args = Config(data_name)
    times = [0, 1, 2, 3, 4]
    seed_nums = [2, 12, 22, 32, 42]

    layer_values = list(args.qlayers) if isinstance(args.qlayers, (list, tuple)) else [args.qlayers]
    experiment_results = {}

    for t, seed_num in zip(times, seed_nums):
        print(f"********************Runtime == {t}***********************")
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)

        for layer_num in layer_values:
            print(f"********** QLAYERS == {layer_num} **********")
            args.qlayers = layer_num
            report_, f1_dev, model_name, train_loss, trues, preds, running_time = main(
                data_name, t=t, seed_num=seed_num, args=args
            )

            result_dir = os.path.join("res", data_name)
            os.makedirs(result_dir, exist_ok=True)
            result_prefix = os.path.join(
                result_dir,
                f"{model_name}_time_{t}_{args.enc}_seed_{seed_num}",
            )
            save_pickle(result_prefix + "train_loss.pickle", train_loss)
            save_pickle(result_prefix + "_reportDic.pickle", report_)
            save_pickle(result_prefix + "_pred.pickle", [trues, preds])

            metrics = experiment_results.setdefault(
                model_name,
                {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "epoch_time": [],
                },
            )
            metrics["accuracy"].append(report_["accuracy"])
            metrics["precision"].append(report_["macro avg"]["precision"])
            metrics["recall"].append(report_["macro avg"]["recall"])
            metrics["f1"].append(report_["macro avg"]["f1-score"])
            metrics["epoch_time"].append(running_time)

    print("\n========== Multiple Experiment Results ==========")
    for model_name, metrics in experiment_results.items():
        print(f"\nModel: {model_name}")
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            values = metrics[metric_name]
            values = np.asarray(values, dtype=np.float64)
            std = values.std(ddof=1) if values.size > 1 else 0.0
            print(f"{metric_name:>9}: {values.mean():.4f} ± {std:.4f}")

        epoch_times = np.asarray(metrics["epoch_time"], dtype=np.float64)
        epoch_time_std = epoch_times.std(ddof=1) if epoch_times.size > 1 else 0.0
        print(
            f"avg epoch training time: "
            f"{epoch_times.mean():.3f} ± {epoch_time_std:.3f} s"
        )



