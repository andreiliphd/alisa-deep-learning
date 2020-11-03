import json
import mmh3
import torch
import flask
from flask import Flask, request

sessionStorage = {}

app = Flask(__name__)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.fc4(x)


model = Net()


@app.route("/", methods=['POST'])
def main():
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    response = {
        "version": request.json['version'],
        "session": request.json['session'],
        "response": {
            "end_session": False
        }
    }

    if request.json['session']['new']:
        response['response']['text'] = "Привет подписчикам YouTube канала andreiliphd! Добро пожаловать в навык графовый кот. \
                                Это приложение использует нейронную сеть в \
                                качестве демонстрации. Нейронная сеть не была \
                                натренирована. Используется PyTorch версии " + torch.__version__
        return json.dumps(
            response,
            ensure_ascii=False,
            indent=2
        )

    request_json = request.json
    hash = mmh3.hash(request_json['request']['original_utterance'])
    norm = hash / 4294967295
    norm = torch.tensor(norm)
    output = norm.reshape([1, 1])
    output = model(output)
    argmax = torch.argmax(output, dim=1)
    response['response']['text'] = "Категория классификации " + str(int(argmax))
    return json.dumps(
        response,
        ensure_ascii=False,
        indent=2
    )

@app.route("/health", methods=['GET'])
def health():
    return flask.Response(status=200)
