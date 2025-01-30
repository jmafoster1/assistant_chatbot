#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:10:10 2025

@author: lesya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:28:12 2025

@author: lesya
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer 
# import emoji
import os
import re 
import torch 
from elg import FlaskService
from elg.model import AnnotationsResponse, TextRequest


path_to_dir = 'lesyar/intent_classifier'

# Default confidence threshold, if not overridden by request parameter
DEFAULT_THRESHOLD = float(os.environ.get("INTENT_THRESHOLD", 0.58))


class Multilang_Intent(FlaskService): 
    model = AutoModelForSequenceClassification.from_pretrained(path_to_dir)
    tokenizer = AutoTokenizer.from_pretrained(path_to_dir)
    device = 'cpu'
    label_map = [
            "GREETING",
            "BUG",
            "FEATURE",
            "OUTPUT",
            "EXPLAIN",
            "END"
        ]



    def run_model(self, input_str, threshold=DEFAULT_THRESHOLD, include_attention=True): 
        
        #getting model output
        inputs = self.tokenizer(input_str, truncation_strategy = True, max_length = 512,return_tensors='pt').to(self.device)

        outputs = self.model(**inputs, output_attentions = include_attention, output_hidden_states = True)

        #gettting class prediction scores 
        preds = outputs.logits 
        probs = torch.nn.Sigmoid()(outputs.logits).view(-1)
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= threshold)] = 1
        print(preds)

        pred_probs = probs[torch.where(preds >= threshold)].detach().numpy()
        
        #pred_class = self.label_map[probs]
        pred_classes = [i for (i, v) in zip(self.label_map,list(preds.bool())) if v]
        #final_outputs = [{"score": score, "label": label} for score, label in pred_probs]        

        return pred_classes

        #note: should we return importance scores if the model doesn't detect any class?

    def process_text(self, request):
        threshold = DEFAULT_THRESHOLD
        try:
            threshold = float(request.params["threshold"])
        except:
            # Either no params, no params["threshold"], or the value wasn't a number.
            # In all cases fall back to default.
            pass

        #outputs = self.run_model(request.content, threshold=threshold)
        outputs = self.run_model(request, threshold=threshold)
        print(outputs)
        #return self.convert_outputs(outputs, request.content)
        return self.run_model(outputs, request)
    
flask_service = Multilang_Intent("bert_intent")
app = flask_service.app