"""學習型 AI 的網路：1D 殘差卷積網路沿 34 種牌掃描（相鄰的牌組成順子、搭子），
輸出每個動作的分數（策略頭，不合法的動作由遮罩擋掉）與這局預期輸贏（價值頭，台）。

需要 PyTorch（訓練用的虛擬環境 .venv-train，見 training/README）。
"""
from __future__ import annotations

import torch
from torch import nn

from training.actions import ACTION_COUNT
from training.encoding import CHANNELS

VALUE_SCALE = 10.0
"""價值頭輸出 × 10 = 台；輸贏多在 ±30 台內，縮小後比較好訓練。"""


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class MahjongNet(nn.Module):
    def __init__(self, width: int = 128, blocks: int = 10) -> None:
        super().__init__()
        self.config = {"width": width, "blocks": blocks}
        self.stem = nn.Sequential(nn.Conv1d(CHANNELS, width, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        # 打牌、宣告、槓都是「對某張牌」的動作：每種牌各出 3 個分數；碰吃過胡看整體
        self.tile_head = nn.Conv1d(width, 3, 1)
        self.global_head = nn.Sequential(nn.Flatten(), nn.Linear(width * 34, 256), nn.ReLU())
        self.other_actions = nn.Linear(256, ACTION_COUNT - 3 * 34)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (動作分數 [B, ACTION_COUNT], 預期輸贏 [B]，單位 VALUE_SCALE 台)。"""
        h = self.body(self.stem(x))
        per_tile = self.tile_head(h).flatten(1)  # [B, 3*34]：打牌、宣告、槓依序
        g = self.global_head(h)
        logits = torch.cat([per_tile, self.other_actions(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def save(model: MahjongNet, path, **extra) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), **extra}, path)


def load(path, device: str = "cpu") -> MahjongNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MahjongNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state"])
    return model.to(device).eval()
