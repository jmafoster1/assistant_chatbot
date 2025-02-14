import os
import json
import aiohttp
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("REACT_APP_MY_WEB_HOOK_URL")


def get_feedback_message(email, message, message_type, archive_url=None):
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": message_type,
            },
        },
    ]

    if email:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<mailto:{email}>",
                },
            },
        )

    message = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": message + "\n[Sent via the chatbot assistant]",
        },
    }

    if archive_url:
        message["accessory"] = {
            "type": "image",
            "image_url": archive_url,
            "alt_text": "Problematic image",
        }
    blocks.append(message)

    return {"blocks": blocks}


async def send_to_slack(email, message, message_type, archive_url=None):
    feedback_message = get_feedback_message(email, message, message_type, archive_url)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            API_URL,  # Assuming API_URL is defined elsewhere
            headers={"Content-Type": "application/json"},
            data=json.dumps(feedback_message),
        ) as response:
            if not response.ok:
                raise ValueError(f"Problem connecting to slack API:\n{response}")
            return response.ok
