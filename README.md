# Multilingual chatbot

This repo contains the service that wraps the chatbot built for the VERA project. In this version, the service contains one part:

- a pure-Python Flask app exposing the classifier with an API compatible with the requirements of the [European Language Grid](https://www.european-language-grid.eu)

## Building the Python part

The model is automatically imported from the publicly available Huggingface repository: lesyar/intent_detection.

## Running the model

The application outputs the class, user's input and the model's output message. Only the output message is supposed to be displayed to the user through the interface.
If the class is either the FEATURE, or BUG, or OUTPUT, then class and the user's input should be sent to the Slack channel.

To test the model, you can run:
print(flask_service.process_text("Hi. I believe the image is actually legit."))
The model shoudl output:
('Hi. Can I upload a bult of images at once?', ['GREETING', 'FEATURE', 'EXPLAIN'], 'Thank you for requesting a new functionality. We have sent it to the relevant team and hope to implement it in nearesr future to improve your experience')
Because the model detected "FEATURE", the first and second elements in the list should be sent to Slack channel.
The last element in the output should always be displayed.

In future, if the model detected "EXPLAIN", it would be calling another model for document retrieval, and it can work with a mock message for now.

## Running as a local service
``quart run -p 5001``
