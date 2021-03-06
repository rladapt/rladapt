from flask import Flask, request, jsonify
from apis.dev2 import AdaptModel
import numpy as np
from utils.config import ar_config, pd_config, logk_config
rl_model_server = Flask(__name__)

# config = {
#     **ar_config,
#     'degrade': lambda: 0,
#     'evaluate': lambda: 0,
# }
# agent = AdaptModel((0, 20)) \
#     .add('resolution', (0.1, 1)) \
#     .add('skip', [1, 30], [1, 2, 3, 5, 6, 10, 15, 30])\
#     .add('quant', [0, 50], 1)\
#     .add_config(config) \
#     .add_profile('/home/shen/research/RL/WANStream/ar_profile') \
#     .build() \
#     .load(10)

# agent = AdaptModel([0, 40])\
#     .add('res', [0.15, 1])\
#     .add('skip', [1, 30], [1, 2, 3, 5, 6, 10, 15, 30])\
#     .add('quant', [0, 50], 1)\
#     .add_profile('/home/shen/research/RL/WANStream/pd_profile')\
#     .add_config(pd_config)\
#     .add_degrade(lambda: 1)\
#     .add_evaluate(lambda: 1)\
#     .build()\
#     .load(15)

model = AdaptModel(0, 40)\
    .add('res', (0.15, 1), 'continuous')\
    .add('skip', 7, 'discrete')\
    .add('quant', 50, 'discrete')\
    .add_degrade(lambda: 1) \
    .add_evaluate(lambda: 1)\
    .add_profile('../models/pd') \
    .build()
# a.test(9)
model.load(40)


@rl_model_server.route('/')
def getparameter():
    bd = request.args.get("bd", "")
    if not bd:
        return ""
    bd = float(bd)
    result, _ = model.predict(bd / pd_config['band_norm'])
    return jsonify(result)


if __name__ == "__main__":
    rl_model_server.run('0.0.0.0', 8888)
    # for bd in np.arange(0, 40, 0.2):
    #     (res, framerate, quantization), _ = model.predict(
    #         bd / pd_config['band_norm'])
    #     print("{:.1f}Mbps: res={}, frame={}, quant={}".format(
    #         bd, res, framerate, quantization
    #     ))
