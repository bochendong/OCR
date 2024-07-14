import torch
import torch.nn as nn
import torch.nn.functional as F

class RLAgent(nn.Module):
    def __init__(self, device):
        super(RLAgent, self).__init__()

        self.device = device

        self.encoder_s_input = self._create_encoder(512)
        self.encoder_s_bbox = self._create_encoder(2048, output_size=512)
        self.encoder_r_input = self._create_encoder(512)
        self.encoder_r_bbox = self._create_encoder(2048, output_size=512)
        
        self.encoder_remained = self._create_encoder(512)

        self.step_processor = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def _create_encoder(self, input_size, output_size=512):
        return nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, state, step):
        selected_input_ids = state["selected_input_ids"].float().to(self.device)
        selected_bbox = state["selected_bbox"].float().to(self.device)
        remain_input_ids = state["remain_input_ids"].float().to(self.device)
        remain_bbox = state["remain_bbox"].float().to(self.device)

        step_tensor = torch.tensor([step], dtype=torch.float32, device=self.device).view(-1, 1)
        step_output = self.step_processor(step_tensor)

        selected = self.encoder_s_input(selected_input_ids) + self.encoder_s_bbox(selected_bbox.view(-1)) 
        selected = selected.unsqueeze(1)
        selected = selected + step_output

        step_tensor = torch.tensor([step - 1], dtype=torch.float32, device=self.device).view(-1, 1)
        step_output = self.step_processor(step_tensor)
        remained = self.encoder_r_input(remain_input_ids) + self.encoder_r_bbox(remain_bbox.view(-1))
        remained = remained.unsqueeze(1)
        remained = remained + step_output

        v = self.encoder_remained(remained).transpose(-2, -1)

        attn_logits = torch.matmul(selected, remained.transpose(-2, -1))
        attention = F.softmax(attn_logits, dim=-1)

        score = torch.matmul(attention, v).squeeze()

        return score.view(-1)
