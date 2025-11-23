from torch import nn
from transformers import AutoModel

class ViSpam_Classifier(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', freeze_bert=False, drop=0.3):
        super().__init__()
        self.model_name = "ViSpam_Model" # Used for folder naming in train function
        self.num_classes = 2
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask, category_id=None):
        # Note: category_id arg added to match train_step signature, but ignored
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        x = self.drop(output)
        x = self.fc(x)
        return x