import time
import threading
from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')

current_status = "Initiální stav"


def read_status_from_file():
    global current_status
    while True:
        with open("status.txt", "r") as file:
            new_status = file.read()
            if new_status != current_status:
                current_status = new_status
        time.sleep(5)  # Časový interval, každých 5 sekund


@app.route('/')
def index():
    return render_template('index.html', status=current_status)


if __name__ == '__main__':
    status_reader_thread = threading.Thread(target=read_status_from_file)
    status_reader_thread.daemon = True
    status_reader_thread.start()

    app.run(host='0.0.0.0', port=80)
