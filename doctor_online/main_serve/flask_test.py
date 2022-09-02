# 导入Flask类
from flask import Flask
# 创建一个该类的实例app, 参数为__name__, 这个参数是必需的，
# 这样Flask才能知道在哪里可找到模板和静态文件等东西。
app = Flask(__name__)

# 使用route()装饰器来告诉Flask触发函数的URL
@app.route('/')
def hello_world():
    """请求指定的url后，执行的主要逻辑函数"""
    # 在用户浏览器中显示信息：'Hello, World!'
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
