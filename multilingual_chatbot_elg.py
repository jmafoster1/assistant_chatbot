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
import torch 
from elg import FlaskService


path_to_dir = 'lesyar/intent_classifier'
# Default confidence threshold, if not overridden by request parameter
DEFAULT_THRESHOLD = float(os.environ.get("INTENT_THRESHOLD", 0.58))


class Multilang_Intent(FlaskService): 
    # load tokenizer abd the model from the huggingface repo
    model = AutoModelForSequenceClassification.from_pretrained(path_to_dir)
    tokenizer = AutoTokenizer.from_pretrained(path_to_dir)
    # I am running it on a cpu, but please adjust this of the GPU is needed
    device = 'cpu'
    label_map = [
            "GREETING",
            "BUG",
            "FEATURE",
            "OUTPUT",
            "EXPLAIN",
            "END"
        ]
    # the match-case comes after python 3.10, so I am not sure it will work in production. Also, my model was trained using python 3.9
    # I therefore will replace keep a non-elegant statement here for now, and replace it if we have python 3.10
    
    def check_class(*classes):
        message=""
        if "GREETING" in classes:
            message="Hello. How can I help ypu today? I can send your feedback about our tools or provide explanations about the results."
        elif "BUG" in classes:
            message="Thank you for reporting the problem. We have sent it to our team to fix. Apologies for any inconvenience using our plugin."
        elif "FEATURE" in classes:
            message="Thank you for requesting a new functionality. We have sent it to the relevant team and hope to implement it in nearesr future to improve your experience"
        elif "OUTPUT" in classes:
            message="Thank you for providing feedback regarding our model. We have taken it into account and will use it to improve the performance of our algorithms."
        elif "EXPLAIN" in classes:
            message="Thank you for your questions. We are looking through our documentation to provide the answer to your enquiry..."
        elif "GREETING" in classes:
            message="Thank you for using the assistant chatbot!"
        return message

    def run_model(self, input_str, threshold=DEFAULT_THRESHOLD, include_attention=True): 
        #getting model output
        inputs = self.tokenizer(input_str, truncation_strategy = True, max_length = 512,return_tensors='pt').to(self.device)
        outputs = self.model(**inputs, output_attentions = include_attention, output_hidden_states = True)

        #gettting class prediction scores 
        preds = outputs.logits 
        probs = torch.nn.Sigmoid()(outputs.logits).view(-1)
        # translate the scores for each class into 1 and 0, which is how the training data was presented
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= threshold)] = 1

        # interpreting each class number with the actual class name
        pred_classes = [i for (i, v) in zip(self.label_map,list(preds.bool())) if v]
        
        # Here, I will add the translation of each class to a message the user sees. This can change as we review the interface:
        output=self.check_class(pred_classes[1])
        
        return input_str, pred_classes, output
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
        #return self.convert_outputs(outputs, request.content)
        return outputs
    
flask_service = Multilang_Intent("bert_intent")
app = flask_service.app

# testing examples
# print(flask_service.process_text("Hi. I believe the image is actually legit."))
