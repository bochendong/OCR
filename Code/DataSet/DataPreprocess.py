from PIL import Image
from transformers import AutoProcessor
from transformers import LayoutLMv2Processor
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader

class DataPreprocessor():
    def __init__(self, model = "Fine_tuned"):
        self.dataset = load_dataset("nielsr/funsd")
        self.labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        self.id2label = {v: k for v, k in enumerate(self.labels)}
        self.label2id = {k: v for v, k in enumerate(self.labels)}

        if (model == "Fine_tuned"):
            self.processor = AutoProcessor.from_pretrained("nielsr/layoutlmv2-finetuned-funsd", apply_ocr = False)
        else:
            self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

        self.features = Features({
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(ClassLabel(names=self.labels)),
        })
    
    def get_data_loader(self):
        def preprocess_data(examples):
            images = [Image.open(path).convert("RGB") for path in examples['image_path']]
            words = examples['words']
            boxes = examples['bboxes']
            word_labels = examples['ner_tags']

            encoded_inputs = self.processor(images, words, boxes=boxes, word_labels=word_labels,
                                    padding="max_length", truncation=True)

            return encoded_inputs
        
        train_dataset = self.dataset['train'].map(preprocess_data, batched=True, \
                                                  remove_columns=self.dataset['train'].column_names, \
                                                    features=self.features)
        test_dataset = self.dataset['test'].map(preprocess_data, batched=True, \
                                                remove_columns=self.dataset['test'].column_names, \
                                                    features=self.features)

        train_dataset.set_format(type="torch")
        test_dataset.set_format(type="torch")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader


        
        