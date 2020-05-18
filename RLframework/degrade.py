from image_interface import visiontask2 as visiontask


def video_degrade(action):
    return visiontask(max(0.15, action[0]), 30, action[1], True, int(action[2]), False)


def video_evaluate(bd_last, result):
    bd, f1score = result
    f1score_loss = - abs(f1score - 1) * 10
    bandwidth_loss = - \
        abs(max(0, bd - bd_last)) / bd_last * 1.5
    reward = f1score_loss + bandwidth_loss
    return reward


if __name__ == "__main__":
    pass
