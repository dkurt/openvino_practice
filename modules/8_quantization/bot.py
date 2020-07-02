import os
import numpy as np
import cv2 as cv
import telebot
import argparse
from style_transfer import StyleTransfer

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='Telegram bot token', required=True)
args = parser.parse_args()


xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy_int8.xml')
bin_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy_int8.bin')
model = StyleTransfer(xml_path, bin_path)

bot = telebot.TeleBot(args.token)


def get_image(message):
    fileID = message.photo[-1].file_id
    file = bot.get_file(fileID)
    data = bot.download_file(file.file_path)
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(buf, cv.IMREAD_COLOR)


def send_image(message, img):
    _, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 90])
    bot.send_photo(message.chat.id, buf)


@bot.message_handler(content_types=['photo'])
def process_image(message):

    img = get_image(message)
    stylized = model.process(img)
    send_image(message, stylized)


bot.polling()
