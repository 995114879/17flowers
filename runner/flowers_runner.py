import os
import sys

print(sys.path)

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def training_ml_model():
    from flowers.ml import training
    training.run(
        img_path_dir="../datas/17flowers",
        algo_output_path="./output/flowers/ml/model.pkl"
    )


def predict_ml_model():
    from flowers.ml import predictor

    p = predictor.Predictor(
        algo_path=r'output/flowers/ml/model.pkl'
    )
    r = p.predict(
        img_path=r'../datas/17flowers/c1/image_0001.jpg',
        k=3
    )
    print(r)


def client_ml_model():
    from flowers.ml import shell_client
    shell_client.run(
        algo_path=r'output/flowers/ml/model.pkl'
    )


if __name__ == '__main__':
    # training_ml_model()
    # predict_ml_model()
    client_ml_model()