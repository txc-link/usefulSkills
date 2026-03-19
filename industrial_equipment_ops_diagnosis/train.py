"""
工业设备运维诊断 - 完整训练脚本
支持：数据增强、分布式训练、模型优化、评估指标
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import logging
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import random
import math
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 数据增强 ==============
class TimeSeriesAugmentation:
    """时序数据增强"""

    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    @staticmethod
    def scaling(data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """幅度缩放"""
        factor = np.random.normal(1.0, sigma, (data.shape[0], 1))
        return data * factor

    @staticmethod
    def time_shift(data: np.ndarray, shift_max: int = 5) -> np.ndarray:
        """时间偏移"""
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(data, shift, axis=0)

    @staticmethod
    def magnitude_warp(data: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
        """幅度扭曲"""
        from scipy.interpolate import CubicSpline

        orig_steps = np.linspace(0, data.shape[1] - 1, knot)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], knot))
        warp_steps = np.linspace(0, data.shape[1] - 1, data.shape[1])

        warped = np.zeros_like(data)
        for i in range(data.shape[0]):
            warper = CubicSpline(orig_steps, random_warps[i])
            warped[i] = data[i] * warper(warp_steps)

        return warped

    @staticmethod
    def apply_augmentation(
        data: np.ndarray,
        aug_types: List[str] = ["noise", "scaling", "shift"]
    ) -> List[np.ndarray]:
        """应用增强"""
        augmented = [data]

        if "noise" in aug_types:
            augmented.append(TimeSeriesAugmentation.add_noise(data.copy()))
        if "scaling" in aug_types:
            augmented.append(TimeSeriesAugmentation.scaling(data.copy()))
        if "shift" in aug_types:
            augmented.append(TimeSeriesAugmentation.time_shift(data.copy()))

        return augmented


# ============== 数据集 ==============
class SensorAnomalyDataset(Dataset):
    """传感器时序异常检测数据集"""

    FAULT_TYPES = {
        0: "normal", 1: "bearing_fault", 2: "gear_wear", 3: "motor_failure",
        4: "temperature_overheat", 5: "vibration_exceed", 6: "pressure_anomaly",
        7: "current_fluctuation", 8: "lubrication_failure", 9: "misalignment",
        10: "imbalance", 11: "looseness", 12: "cooling_failure",
        13: "power_supply_issue", 14: "flow_blockage", 15: "leakage", 16: "resonance"
    }

    def __init__(
        self,
        data_path: str,
        window_size: int = 100,
        stride: int = 10,
        split: str = "train",
        augment: bool = False,
        sensor_names: List[str] = None
    ):
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.augment = augment and (split == "train")
        self.sensor_names = sensor_names or ["vibration", "temperature", "current", "pressure"]

        # 加载数据
        self.data = self._load_data(data_path)
        self.samples = self._create_samples()

        # 划分数据集
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(n * 0.7)]
        elif split == "val":
            self.samples = self.samples[int(n * 0.7):int(n * 0.85)]
        else:
            self.samples = self.samples[int(n * 0.85):]

        # 预计算统计量
        self._compute_stats()

    def _load_data(self, path: str) -> list:
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict("records")
        raise ValueError(f"Unsupported format: {path.suffix}")

    def _compute_stats(self):
        """计算统计量用于归一化"""
        all_values = []
        for s in self.samples:
            all_values.append(s["x"])

        all_values = np.concatenate(all_values, axis=0)

        self.mean = np.mean(all_values, axis=0, keepdims=True)
        self.std = np.std(all_values, axis=0, keepdims=True)
        self.std[self.std == 0] = 1

    def _create_samples(self) -> list:
        samples = []
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            window = self.data[i:i + self.window_size]

            # 提取特征
            x = np.array([
                [d.get(s, 0) for s in self.sensor_names]
                for d in window
            ], dtype=np.float32)

            # 标签
            anomaly_label = window[-1].get("anomaly_label", 0)
            type_label = window[-1].get("type_label", 0)
            severity_label = window[-1].get("severity_label", 0)  # 0-3

            samples.append({
                "x": x,
                "anomaly_label": anomaly_label,
                "type_label": type_label,
                "severity_label": severity_label
            })

        return samples

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample["x"].copy()

        # 增强
        if self.augment and random.random() > 0.5:
            aug_type = random.choice(["noise", "scaling", "shift"])
            if aug_type == "noise":
                x = TimeSeriesAugmentation.add_noise(x, 0.02)
            elif aug_type == "scaling":
                x = TimeSeriesAugmentation.scaling(x, 0.1)
            else:
                x = TimeSeriesAugmentation.time_shift(x, 3)

        x = self._normalize(x)

        return (
            torch.FloatTensor(x),
            torch.LongTensor([sample["anomaly_label"]]),
            torch.LongTensor([sample["type_label"]]),
            torch.LongTensor([sample["severity_label"]])
        )


# ============== 分布式采样器 ==============
class ImbalancedDatasetSampler(Sampler):
    """不平衡数据集采样器"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        # 统计各类型数量
        label_counts = defaultdict(int)
        for i in self.indices:
            label = dataset.samples[i]["anomaly_label"]
            label_counts[label] += 1

        # 计算权重
        weights = []
        for i in self.indices:
            label = dataset.samples[i]["anomaly_label"]
            weight = 1.0 / label_counts[label]
            weights.append(weight)

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.indices), replacement=True).tolist())

    def __len__(self):
        return len(self.indices)


# ============== 数据生成器 ==============
def generate_synthetic_data(
    num_samples: int = 10000,
    anomaly_ratio: float = 0.15,
    output_path: str = "data/sensor_data.json"
):
    """生成模拟传感器数据"""
    np.random.seed(42)
    random.seed(42)

    FAULT_TYPES = list(range(1, 17))

    data = []
    anomaly_count = int(num_samples * anomaly_ratio)
    anomaly_indices = set(random.sample(range(num_samples), anomaly_count))

    for i in range(num_samples):
        sample = {
            "timestamp": f"2025-03-{(i // 1440) + 1:02d}T{i % 1440 // 60:02d}:{i % 60:02d}:00",
            "vibration": np.random.normal(5.0, 1.0),
            "temperature": np.random.normal(50.0, 5.0),
            "current": np.random.normal(10.0, 1.0),
            "pressure": np.random.normal(100.0, 10.0),
            "anomaly_label": 0,
            "type_label": 0,
            "severity_label": 0
        }

        if i in anomaly_indices:
            fault_type = random.choice(FAULT_TYPES)
            severity = random.choice([1, 2, 3])  # low, medium, high

            sample["anomaly_label"] = 1
            sample["type_label"] = fault_type
            sample["severity_label"] = severity

            # 注入异常特征
            if fault_type == 1:  # bearing_fault
                sample["vibration"] += random.uniform(8, 15)
                sample["temperature"] += random.uniform(10, 20)
            elif fault_type == 4:  # temperature_overheat
                sample["temperature"] += random.uniform(20, 35)
            elif fault_type == 5:  # vibration_exceed
                sample["vibration"] += random.uniform(10, 20)
            elif fault_type == 6:  # pressure_anomaly
                sample["pressure"] += random.uniform(-30, -15)
            elif fault_type == 7:  # current_fluctuation
                sample["current"] += random.uniform(-8, 8)

        data.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {num_samples} samples, {anomaly_count} anomalies")
    return data


# ============== 训练器 ==============
class AnomalyDetectorTrainer:
    """异常检测模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_amp: bool = True,  # 混合精度
        gradient_accumulation: int = 1
    ):
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation = gradient_accumulation

        # 多GPU支持
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        self.model = model.to(device)

        # 损失函数 - 类别权重
        self.anomaly_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 3.0]).to(device)  # 异常样本权重更高
        )
        self.type_criterion = nn.CrossEntropyLoss()
        self.severity_criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,
            steps_per_epoch=100,
            pct_start=0.1
        )

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 统计
        self.history = defaultdict(list)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            x, anomaly_labels, type_labels, severity_labels = batch
            x = x.to(self.device)
            anomaly_labels = anomaly_labels.squeeze().to(self.device)
            type_labels = type_labels.squeeze().to(self.device)
            severity_labels = severity_labels.squeeze().to(self.device)

            # 混合精度前向
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
                    anomaly_logits, type_logits, severity_logits = outputs[:3]

                    # 损失
                    anomaly_loss = self.anomaly_criterion(anomaly_logits, anomaly_labels)

                    # 仅对异常样本计算类型和严重度损失
                    anomaly_mask = (anomaly_labels == 1)
                    if anomaly_mask.sum() > 0:
                        type_loss = self.type_criterion(
                            type_logits[anomaly_mask],
                            type_labels[anomaly_mask]
                        )
                        severity_loss = self.severity_criterion(
                            severity_logits[anomaly_mask],
                            severity_labels[anomaly_mask]
                        )
                        type_loss = type_loss * 0.5
                        severity_loss = severity_loss * 0.3
                    else:
                        type_loss = 0
                        severity_loss = 0

                    loss = anomaly_loss + type_loss + severity_loss
                    loss = loss / self.gradient_accumulation
            else:
                outputs = self.model(x)
                anomaly_logits, type_logits, severity_logits = outputs[:3]

                anomaly_loss = self.anomaly_criterion(anomaly_logits, anomaly_labels)

                anomaly_mask = (anomaly_labels == 1)
                if anomaly_mask.sum() > 0:
                    type_loss = self.type_criterion(
                        type_logits[anomaly_mask],
                        type_labels[anomaly_mask]
                    )
                    severity_loss = self.severity_criterion(
                        severity_logits[anomaly_mask],
                        severity_labels[anomaly_mask]
                    )
                else:
                    type_loss = 0
                    severity_loss = 0

                loss = anomaly_loss + type_loss * 0.5 + severity_loss * 0.3
                loss = loss / self.gradient_accumulation

            # 反向
            if self.use_amp:
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation

            # 准确率
            preds = torch.argmax(anomaly_logits, dim=-1)
            correct += (preds == anomaly_labels).sum().item()
            total += len(anomaly_labels)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        total_loss = 0
        correct = 0
        total = 0

        # 混淆矩阵统计
        tp = fp = tn = fn = 0

        for batch in dataloader:
            x, anomaly_labels, type_labels, severity_labels = batch
            x = x.to(self.device)
            anomaly_labels = anomaly_labels.squeeze().to(self.device)

            outputs = self.model(x)
            anomaly_logits = outputs[0]

            loss = self.anomaly_criterion(anomaly_logits, anomaly_labels)
            total_loss += loss.item()

            probs = torch.softmax(anomaly_logits, dim=-1)[:, 1]
            preds = torch.argmax(anomaly_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(anomaly_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (preds == anomaly_labels).sum().item()
            total += len(anomaly_labels)

            # 混淆矩阵
            tp += ((preds == 1) & (anomaly_labels == 1)).sum().item()
            fp += ((preds == 1) & (anomaly_labels == 0)).sum().item()
            tn += ((preds == 0) & (anomaly_labels == 0)).sum().item()
            fn += ((preds == 0) & (anomaly_labels == 1)).sum().item()

        # 计算指标
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stop_patience: int = 10,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        best_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # 验证
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

            # 保存最佳模型
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(
                    Path(checkpoint_dir) / "best_model.pt",
                    epoch,
                    val_metrics
                )
                patience_counter = 0
                print(f"  -> Best model saved! F1: {best_f1:.4f}")
            else:
                patience_counter += 1

            # 保存最后一个
            if epoch % 10 == 0:
                self.save_checkpoint(
                    Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt",
                    epoch,
                    val_metrics
                )

            # 早停
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            print()

        return dict(self.history)

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        model_state = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": dict(self.history)
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.model, DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = defaultdict(list, checkpoint.get("history", {}))
        return checkpoint.get("epoch", 0)


# ============== 模型优化 ==============
class ModelOptimizer:
    """模型优化工具"""

    @staticmethod
    def quantize(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """动态量化"""
        return torch.quantization.quantize_dynamic(
            model,
            {nn.LSTM, nn.Linear},
            dtype=dtype
        )

    @staticmethod
    def prune(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """结构化剪枝"""
        import torch.nn.utils.prune as prune

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")

        return model

    @staticmethod
    def export_onnx(model: nn.Module, input_shape: tuple, output_path: str):
        """导出 ONNX"""
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["anomaly", "type", "severity"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "anomaly": {0: "batch_size"},
                "type": {0: "batch_size"},
                "severity": {0: "batch_size"}
            }
        )
        print(f"Exported to ONNX: {output_path}")


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="Train Industrial Anomaly Detector")
    parser.add_argument("--data_path", type=str, default="data/sensor_data.json")
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    # 生成数据
    if args.generate_data:
        generate_synthetic_data(args.num_samples, output_path=args.data_path)

    # 导入模型
    from skill import TemporalAnomalyDetector

    # 数据集
    train_dataset = SensorAnomalyDataset(
        args.data_path, args.window_size, args.stride, "train", args.augment
    )
    val_dataset = SensorAnomalyDataset(
        args.data_path, args.window_size, args.stride, "val", False
    )

    # 采样器 - 处理不平衡
    train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 模型
    model = TemporalAnomalyDetector(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    # 训练器
    trainer = AnomalyDetectorTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        use_amp=args.use_amp
    )

    # 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )

    # 导出 ONNX
    if args.export_onnx:
        ModelOptimizer.export_onnx(
            model,
            (1, 100, 4),
            f"{args.checkpoint_dir}/model.onnx"
        )

    # 保存历史
    with open(Path(args.checkpoint_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
