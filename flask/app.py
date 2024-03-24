from flask import Flask, request, render_template, redirect, url_for
from base64 import b64encode
import pika

app = Flask(__name__)
image = b''

def send_message_to_rabbitmq(message):
    connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1', 5672, credentials=pika.PlainCredentials('test', 'test')))
    channel = connection.channel()

    channel.queue_declare(queue='message_queue')

    channel.basic_publish(exchange='',
                          routing_key='message_queue',
                          body=message)
    connection.close()

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form.get('message')
    send_message_to_rabbitmq(message)
    # Clear the global image variable
    global image
    image = b''
    # Redirect to the receive_image route
    return redirect(url_for('receive_image'))

@app.route('/receive_image', methods=["POST"])
def receive_image():
    global image
    if request.method == "POST":
        received_image = request.files['image']
        image = received_image.read()
    return render_template(
        "index.html",
        mimetype='image/png', 
        image=b64encode(image).decode('ascii'),
    )

@app.route('/receive_image', methods=['GET'])
def display_image():
    global image
    if not image:
        return "No image has been uploaded yet.", 404
    return render_template(
        "index.html",
        mimetype='image/png', 
        image=b64encode(image).decode('ascii'),
    )

if __name__ == '__main__':
    app.run(port=5002, debug=True)