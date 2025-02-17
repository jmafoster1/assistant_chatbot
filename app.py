from quart import Quart
from multilingual_chatbot_elg import Multilang_Intent
from llama_rag import LLAMA_RAG

app = Quart(__name__)
model = Multilang_Intent("bert_intent")

llama_rag = LLAMA_RAG()


@app.route("/chat/<string:query>", methods=["GET", "POST"])
async def json(query):
    _, pred_classes, output = model.run_model(query)
    if "EXPLAIN" in pred_classes:
        output, docs = llama_rag.call_rag(query)
    return {"userMessageClasses": pred_classes, "message": output, "status": "success"}
