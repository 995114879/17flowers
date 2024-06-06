import os
import sys

print(sys.path)

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def tt_ml_model():
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

    # training_ml_model()
    # predict_ml_model()
    client_ml_model()


def tt_dl_model_v0():
    def training_model():
        from flowers.dl_v0 import training
        training.run(
            img_path_dir="../datas/17flowers",
            total_epoch=3, batch_size=8,
            model_output_path="./output/flowers/dl_v0/model.pkl"
        )

    def predict_model():
        from flowers.dl_v0 import predictor

        p = predictor.Predictor(
            algo_path=r'output/flowers/dl_v0/model.pkl'
        )
        r = p.predict(
            # img_path=r'../datas/17flowers/c1/image_0001.jpg',
            img_path=r'../datas/17flowers/c9/image_0706.jpg',
            # img_path=r'../datas/17flowers/c8/image_0561.jpg',
            k=3
        )
        print(r)

    def client_model_with_shell():
        from flowers.dl_v0 import shell_client
        shell_client.run(
            algo_path=r'output/flowers/dl_v0/model.pkl'
        )

    # training_model()
    # predict_model()
    client_model_with_shell()

def tt_dl_model_v1():
    def training_model():
        from flowers.dl_v1 import training
        training.run(
            img_path_dir="../datas/17flowers",
            total_epoch=3, batch_size=8,
            model_output_path="./output/flowers/dl_v1/model.pkl"
        )

    def predict_model():
        from flowers.dl_v1 import predictor

        p = predictor.Predictor(
            algo_path=r'output/flowers/dl_v1/model.pkl'
        )
        r = p.predict(
            # img_path=r'../datas/17flowers/c1/image_0001.jpg',
            img_path=r'../datas/17flowers/c9/image_0706.jpg',
            # img_path=r'../datas/17flowers/c8/image_0561.jpg',
            k=3
        )
        print(r)

    def client_model_with_shell():
        from flowers.dl_v1 import shell_client
        shell_client.run(
            algo_path=r'output/flowers/dl_v1/model.pkl'
        )

    def start_flask_api():
        from flowers.dl_v1 import flask_app
        app.run()

    # training_model()
    # predict_model()
    # client_model_with_shell()
    start_flask_api()




if __name__ == '__main__':
    # tt_ml_model()
    # tt_dl_model_v0()
    tt_dl_model_v1()
