from flask import Flask
import random
app = Flask(__name__)


@app.route('/<int:size>')
def streaming(size):
    return chr(random.randint(97, 97+25)) * (size // 8)


if __name__ == "__main__":
    app.run('127.0.0.1', 8888)
