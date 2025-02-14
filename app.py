from quart import Quart, request


from multilingual_chatbot_elg import Multilang_Intent
from slack_integration import send_to_slack

app = Quart(__name__)
model = Multilang_Intent("bert_intent")


@app.route("/chat/", methods=["POST"])
async def json():
    data = await request.get_json()  # Get JSON data from the request body
    print("DATA", data)
    message = data.get("message")
    email = data.get("email")
    archive_url = data.get("archiveURL")
    _, pred_classes, output = model.run_model(message)
    if "BUG" in pred_classes:
        await send_to_slack(email, message, "BUG", archive_url)
    if "FEATURE" in pred_classes:
        await send_to_slack(email, message, "FEATURE", archive_url)
    if "OUTPUT" in pred_classes:
        await send_to_slack(email, message, "IMPROVEMENT", archive_url)
    if "EXPLAIN" in pred_classes:
        print("TODO: Hook up to LLM")
    return {"userMessageClasses": pred_classes, "message": output, "status": "success"}
