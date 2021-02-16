import numpy as np
import matplotlib.pyplot as plt
from flask import make_response, Blueprint
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import file


app = Blueprint('app_module', __name__)


# 円グラフを描画
@app.route("/graph.png")
def func():

    All_index = len(file.Neu_list)

    Neu_per = file.Neu_count / All_index
    Pos_per = file.Pos_count / All_index
    Neg_per = file.Neg_count / All_index

    # グラフのimg化
    x = np.array([Pos_per, Neu_per, Neg_per])
    fig1, ax1 = plt.subplots()
    ax1.pie(x)

    canvas = FigureCanvasAgg(fig1)
    buf = io.BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()

    response = make_response(data)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Length'] = len(data)
    return response

