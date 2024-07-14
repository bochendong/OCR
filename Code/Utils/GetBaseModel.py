from transformers import AutoModelForTokenClassification, LayoutLMv2ForTokenClassification


def getBaseModel(id2label, label2id, model = "Fine_tune"):
    if (model == "Fine_tune"):
        model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")
    elif(model == "LayoutLMv3"):
        model = AutoModelForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")
    else:
        model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(label2id))
        
    model.config.id2label = id2label
    model.config.label2id = label2id  

    return model
