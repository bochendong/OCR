import torch
import math

class Env(object):
    def __init__(self, model, device, model_type = "fine-tune"):
        self.model_type = model_type
        self.OCR_Model = model
        self.device = device

    def reset(self, batch):
        device = self.device
        batch = {k: v.squeeze() for k, v in batch.items()}
        
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        target = batch['labels'].to(device)
        if (self.model_type != "LayoutLMv3"):
            image = batch['image'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            remain_token_type_ids = token_type_ids.clone()
            selected_token_type_ids = torch.zeros_like(token_type_ids).to(device)

        else:
            image = batch['pixel_values'].to(device)

        attention_mask = batch['attention_mask'].to(device)
        mask = torch.zeros_like(batch['input_ids'], dtype=torch.bool).to(device)

        remain_input_ids = input_ids.clone()
        remain_bbox = bbox.clone()
        remain_target = target.clone()

        selected_input_ids = torch.zeros_like(input_ids).to(device)
        selected_bbox = torch.zeros_like(bbox).to(device)
        selected_target = torch.full((512,), -100).to(device)

        selected_input_ids[0] = 101 # <CLS Token>
        selected_target[0] = -100
        if (self.model_type == "LayoutLMv3"):
            self.state = {
                        "selected_input_ids": selected_input_ids, "selected_bbox": selected_bbox,
                        "selected_target": selected_target, "remain_input_ids": remain_input_ids, 
                        "remain_bbox": remain_bbox, "remain_target": remain_target,
                        "attention_mask": attention_mask, "mask": mask, "image": image, 
                        "step" : 1
                    }

        else:
            self.state = {
                        "selected_input_ids": selected_input_ids, "selected_bbox": selected_bbox,
                        "selected_token_type_ids": selected_token_type_ids,  "selected_target": selected_target,
                        "remain_token_type_ids": remain_token_type_ids, "remain_input_ids": remain_input_ids, 
                        "remain_bbox": remain_bbox, "remain_target": remain_target,
                        "attention_mask": attention_mask, "mask": mask, "image": image, 
                        "step" : 1
                    }
        
        return self.state

    def get_result(self, state, type = "RL"):
        if (type == "RL"):
            input_ids = state["selected_input_ids"]
            bbox = state["selected_bbox"]
            target = state["selected_target"]
            image = state["image"]
            mask = state["mask"]
            if (self.model_type != "LayoutLMv3"):
                token_type_ids = state["selected_token_type_ids"]
        else:
            input_ids = state["remain_input_ids"]
            bbox = state["remain_bbox"]
            target = state["remain_target"]
            image = state["image"]
            mask = state["attention_mask"]
            if (self.model_type != "LayoutLMv3"):
                token_type_ids = state["remain_token_type_ids"]

        if (self.model_type == "LayoutLMv3"):
            outputs = self.OCR_Model(
                input_ids = input_ids.unsqueeze(0),
                attention_mask = mask.unsqueeze(0),
                bbox = bbox.unsqueeze(0),
                pixel_values = image.unsqueeze(0),
                labels = target
            )
        else:
            outputs = self.OCR_Model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
                token_type_ids=token_type_ids.unsqueeze(0),
                bbox=bbox.unsqueeze(0),
                image=image.unsqueeze(0),
                labels=target
            )


        return outputs

    def reward(self, state):
        outputs = self.get_result(state)

        cur_loss = outputs.loss.item()
        if (math.isnan(cur_loss)):
            reward = 0
        else:
            reward = -cur_loss
        
        return reward, cur_loss
        
    def update(self, actions):
        reward = 0
        state = self.state
        if (self.model_type != "LayoutLMv3"):
            image, attention_mask = state["image"], state["attention_mask"]
            remain_bbox, selected_bbox = state["remain_bbox"], state["selected_bbox"]
            selected_target, remain_target = state["selected_target"], state["remain_target"]
            remain_input_ids, selected_input_ids = state["remain_input_ids"], state["selected_input_ids"]
            selected_token_type_ids, remain_token_type_ids = state["selected_token_type_ids"], state["remain_token_type_ids"]
        else:
            image, attention_mask = state["pixel_values"], state["attention_mask"]
            remain_bbox, selected_bbox = state["remain_bbox"], state["selected_bbox"]
            selected_target, remain_target = state["selected_target"], state["remain_target"]
            selected_token_type_ids, remain_token_type_ids = state["selected_token_type_ids"], state["remain_token_type_ids"]

        step = state["step"]

        for action in actions:
            if remain_input_ids[action] == 0:
                reward -= 1
            selected_input_ids[step] = remain_input_ids[action]
            selected_token_type_ids[step] = remain_token_type_ids[action]
            selected_bbox[step] = remain_bbox[action]
            selected_target[step] = remain_target[action]

            mask = selected_input_ids != 0

            remain_input_ids[action] = 0
            remain_token_type_ids[action] = 0
            remain_bbox[action] = torch.tensor([0, 0, 0, 0])
            remain_target[action] = -100

            step += 1

        selected_input_ids[step + 1] = 102  # <EOS> Token
        selected_target[step + 1] = -100

        if (self.model_type != "LayoutLMv3"):
            self.state = {
                        "selected_input_ids": selected_input_ids, "selected_bbox": selected_bbox,
                        "selected_token_type_ids": selected_token_type_ids,  "selected_target": selected_target,
                        "remain_token_type_ids": remain_token_type_ids, "remain_input_ids": remain_input_ids, 
                        "remain_bbox": remain_bbox, "remain_target": remain_target,
                        "attention_mask": attention_mask, "mask": mask, "image": image, "step" : step
                    }
        else:
            self.state = {
                        "selected_input_ids": selected_input_ids, "selected_bbox": selected_bbox,
                        "selected_target": selected_target, "remain_input_ids": remain_input_ids, 
                        "remain_bbox": remain_bbox, "remain_target": remain_target,
                        "attention_mask": attention_mask, "mask": mask, "image": image, "step" : step
                    }

        env_reward, cur_loss = self.reward(self.state)

        reward = reward + env_reward
        
        return self.state, reward, cur_loss