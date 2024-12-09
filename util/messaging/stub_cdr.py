from re import M
from flask import Flask, request, jsonify
import threading
import requests

from schema.cdr_schemas.events import Event, MapEventPayload

app = Flask(__name__)

# Store registered webhooks
registered_webhooks = []


@app.route("/user/me/registrations", methods=["GET"])
def get_registrations():
    return jsonify({})


@app.route("/user/me/register", methods=["POST"])
def register():
    data = request.get_json()
    webhook_secret = data["webhook_secret"]
    callback_url = data["callback_url"]

    if webhook_secret != "maps rock":
        return jsonify({"error": "Invalid webhook secret"}), 401

    if callback_url:
        registered_webhooks.append(callback_url)
        return jsonify({"id": "4567"}), 200
    else:
        return jsonify({"error": "URL is required"}), 400


def call_webhook(url):
    try:
        map_event = Event(id="1234", event="ping", payload={})
        response = requests.post(url, json=map_event.model_dump())
        print(f"Called webhook {url} with status code {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to call webhook {url}: {e}")


@app.route("/trigger", methods=["POST"])
def trigger_webhooks():
    for url in registered_webhooks:
        threading.Thread(target=call_webhook, args=(url,)).start()
    return jsonify({"message": "triggering webhooks"}), 200


if __name__ == "__main__":
    app.run(port=5050)
