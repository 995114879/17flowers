from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask Demo"

@app.route("/test")
def test():
    return "test Demo"

@app.route("/login/<user>/<password>")
@app.route("/login", methods=["GET", "POST"])
def login(user=None, password=None):
    """
    模拟一个登录逻辑，如果用户名=小明，密码=123456，那么返回True，表示登陆成功
    :return:
    """
    if user is None or password is None:
        if request.method == 'GET':
            _args = request.args  # 获取GET请求的参数字典对象
        else:
            _args = request.form  # 获取POST请求的参数字典对象
        user = _args.get("user")
        password = _args.get("password")
    print(f"user:{user}, password:{password}")
    if "xiaoming" == user and "123456" == password:
        return jsonify({
            'code': 0,
            'msg': '登录成功',
            'user': user,
            'passwd': password
        })
    else:
        return jsonify({
            'code': 1,
            'msg': '登录失败',
            'user': user,
            'passwd': password
        })


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",  # 监听的IP地址，0.0.0.0表示当前机器的所有IP地址均监听
        port=9001,  # 监听的端口号
        debug=True  # debug=True表示当代码文件发生变化的时候，重新启动应用
    )