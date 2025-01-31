from quart import Quart
from multilingual_chatbot_elg import Multilang_Intent

app = Quart(__name__)
model = Multilang_Intent("bert_intent")


@app.route("/chat/<string:message>", methods=["GET", "POST"])
async def json(message):
    _, pred_classes, output = model.run_model(message)
    return {"userMessageClasses": pred_classes, "message": output, "status": "success"}
