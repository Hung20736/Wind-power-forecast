from flask import Flask, flash, render_template, request, redirect
from fastai.tabular import *


server = Flask(__name__)

path = Path(__file__).parent

@server.route('/')
def index():
    return render_template('index.html')

def create_inference_learner(test, type):
    procs = [FillMissing, Normalize]
    cont_names = ["Air temperature | ('C)",	"Pressure | (atm)",	"Wind direction | (deg)",	"Wind speed | (m/s)"]
    test_list = TabularList.from_df(test, cont_names=cont_names, procs=procs)
    learner = load_learner(path/'models', f'{type}.pkl', test_list)
    return learner

@server.route('/', methods=['POST'])
def predict():
    temperature = float(request.form['Air_temperature'])
    pressure = float(request.form['Pressure'])
    direction = float(request.form['Wind_direction'])
    speed = float(request.form['Wind_speed'])
    number = int(request.form['Number'])
    data = {"Air temperature | ('C)":temperature, "Pressure | (atm)":pressure, "Wind direction | (deg)":direction, "Wind speed | (m/s)":speed}
    test = pd.DataFrame(data,index=[0])
    learner = create_inference_learner(test, request.form["type"])
    pred, target = learner.get_preds(DatasetType.Test)
    return render_template('response.html', pred=float(pred)*number)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@server.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        test = pd.read_csv(request.files['file'])
        number = int(request.form['Number'])
        learner = create_inference_learner(test, request.form['type'])
        pred, target = learner.get_preds(DatasetType.Test)
        pred = pred.flatten().tolist()
        pred = [float(x)*number for x in pred]
        return render_template('response.html', pred= pred)
        

if __name__ == '__main__':
    server.run(debug=True)