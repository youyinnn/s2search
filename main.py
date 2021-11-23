from flask import Flask,jsonify
import json
from query_module import query_from_API



app = Flask(__name__)
json_rlt = ""

@app.route('/')
def hello_world():
    return 'hello, welcome to iRanking system of COEN 6313'


@app.route('/search/<string:keyword>&<string:number>', methods=['GET','POST'])
def query_result_req(keyword,number):
    """
    :param keyword: keyword you wanna search like "cloud computing"
    :param number: how many papers you wanna return with the highest rank
    :return: Json list
    """
    '''when debug = True, will not call s2search module, which includes the whole machine learning predict model and 
    complex env requirements, only feed an example of paper dict that for debug web funtion.'''
    paper_list,score = query_from_API(keyword,number,keyword,debug=True)

    return jsonify(paper_list)


if __name__ == '__main__':
    app.run(debug=True)

