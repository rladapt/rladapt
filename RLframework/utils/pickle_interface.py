# operations for pickle file
import pickle


def read(path):
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content


def save(path, content):
    try:
        with open(path, 'wb') as f:
            pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)
        return False
    return True


def savecsv(x, y, path, xmin=1, ymin=0.6):
    '''
    save as latex data point 
        x,y
        1,2
        3,4
    '''
    assert len(x) == len(y)
    with open(path, 'w') as f:
        f.write('x y\n')
        for i in range(len(x)):
            if y[i] >= ymin and x[i] >= xmin:
                f.write("{} {}\n".format(str(x[i]), str(y[i])))


if __name__ == "__main__":
    print(read('/home/shen/research/RL/WANStream/profile/awstream.pickle'))
