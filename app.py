from quart import Quart, request
from multilingual_chatbot_elg import Multilang_Intent
from llama_rag import LLAMA_RAG

app = Quart(__name__)
model = Multilang_Intent("bert_intent")

llama_rag = LLAMA_RAG()


@app.route("/chat/", methods=["POST"])
async def json():
    data = await request.get_json()  # Get JSON data from the request body
    message = data.get("message")
    _, pred_classes, output = model.run_model(message)
    if "EXPLAIN" in pred_classes:
        output, _ = llama_rag.call_rag(message)
    return {
        "userMessageClasses": ["IMPROVEMENT" if p == "OUTPUT" else p for p in pred_classes],
        "message": output,
        "status": "success",
    }
