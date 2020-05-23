def bandwidth_reward(env, bd):
    return -abs((bd - env) / max(bd, env))


def accuracy_reward(accuracy):
    return -abs(accuracy - 1)
