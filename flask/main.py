from flask import Flask, render_template
import CircleGraph, file, html_parser

app = Flask(__name__)

app.register_blueprint(CircleGraph.app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/surveyed', methods=["GET"])
def surveyed():
    return render_template('surveyed.html', className=file.className_list, table=html_parser.soup,
                           Neu=file.Neu_count, Pos=file.Pos_count, Neg=file.Neg_count, title=file.csv_file_path)


@app.route('/mypage')
def mypage():
    return render_template('mypage.html')


if __name__ == '__main__':
    app.run(debug=True)
