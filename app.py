from flask import Flask, request, render_template, jsonify
# from model import predict_result
import pandas as pd
import difflib
app = Flask(__name__)


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/like', methods=['GET','POST'])
def like():
    music_name = request.form.get('name')
    print(music_name)
    user_dict = {}
    # print('1111111')
    with open('user_feedback.txt', 'r', encoding = 'utf8') as fr:
        # print('2222222')
        lines = fr.readlines()
        # print(lines)
        for line in lines:
            # print('33')
            line = line.strip()
            name = ''
            if len(line.split('\t'))>0:
                name = line.split('\t')[0]
            score = 0.0
            if len(line.split('\t'))>1:
                score = line.split('\t')[1]
            # print(string_similar(name, music_name))
            if string_similar(name, music_name)>=0.8:
                user_dict[name] = float(score)+0.5
                # print('into........')
            else:
                user_dict[name] = float(score)

    # result = predict_result.predict(words)
    # return jsonify({'result':result})
    with open('user_feedback.txt', 'w', encoding = 'utf8') as fw:
        for user in user_dict:
            fw.write(user+'\t'+str(user_dict[user])+'\n')
    return jsonify({'result':'will optimize according to your feedback'})


@app.route('/dislike', methods=['GET','POST'])
def dislike():
    music_name = request.form.get('name')
    print(music_name)
    user_dict = {}
    with open('user_feedback.txt', 'r', encoding = 'utf8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            name = ''
            if len(line.split('\t'))>0:
                name = line.split('\t')[0]
            score = 0.0
            if len(line.split('\t'))>1:
                score = line.split('\t')[1]
            # print(string_similar(name, music_name))
            if string_similar(name, music_name)>=0.8:
                user_dict[name] = float(score)-0.5
                # print('into........')
            else:
                user_dict[name] = float(score)

    # result = predict_result.predict(words)
    # return jsonify({'result':result})
    with open('user_feedback.txt', 'w', encoding = 'utf8') as fw:
        for user in user_dict:
            fw.write(user+'\t'+str(user_dict[user])+'\n')
    return jsonify({'result':'will optimize according to feedback'})

@app.route('/predict', methods=['GET','POST'])
def predict():
    age = request.args.get('age')
    HR = request.args.get('HR')

    min_value = int(HR.split('-')[0])
    max_value = int(HR.split('-')[1].split('%')[0])
    # print(min_value)
    # print(max_value)

    user_dict = {}
    with open('user_feedback.txt', 'r', encoding = 'utf8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            name = ''
            # print(line)
            if len(line.split('\t'))>0:
                name = line.split('\t')[0]
            # print(name)
            score = 0.0
            if len(line.split('\t'))>1: 
                score = line.split('\t')[1]
            user_dict[name] = score
    print(user_dict)

    data = pd.read_csv('data.csv', encoding='gbk')
    similar_list = {}
    avg = (min_value+max_value)/2
    for i in range(data.shape[0]):
        current_value = data.iloc[i,-2]
        for user in user_dict:
            # print(data.iloc[i,0])
            # print(user)
            if string_similar(data.iloc[i,0], user)>=0.8:
                if data.iloc[i,0] in user_dict:
                    similar_list[data.iloc[i,0]] = abs(current_value-avg)-float(user_dict[data.iloc[i,0]])
                else:
                    similar_list[data.iloc[i,0]] = abs(current_value-avg)
    sorted_list = sorted(similar_list.items(), key=lambda d: d[1])
    # print(sorted_list)
    num = 0
    music_list = []
    for name in sorted_list:
        if num>=10:
            break
        # print
        music_list.append({'name':name[0]})
        num += 1
    print(music_list)
    # result = predict_result.predict(words)
    return jsonify({'result':music_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
