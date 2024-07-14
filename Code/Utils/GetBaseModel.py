from transformers import AutoModelForTokenClassification, LayoutLMv2ForTokenClassification


def getBaseModel(id2label, label2id, model = "Fine_tuned"):
    if (model == "Fine_tuned"):
        print("layoutlmv2 fine tune model used")
        model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
    elif(model == "LayoutLMv3"):
        print("layoutlmv3 fine tune model used")
        model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
    else:
        print("layoutlmv2 base model used")
        model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(label2id))
        
    model.config.id2label = id2label
    model.config.label2id = label2id  

    return model
