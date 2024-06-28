
### Note: pip/conda install ultralytics before running this notebook


import imaplib
import email
import io
import time
import requests
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms, InterpolationMode
from ultralytics import YOLO
import timm
import yaml
import logging
import re

# Suppress YOLO logs
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Load configuration from YAML file
def load_config():
    with open('../config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

IMAP_HOST = config['imap_config']['host']
EMAIL_USER = config['imap_config']['user']
EMAIL_PASS = config['imap_config']['password']
TELEGRAM_BOT_TOKEN = config['telegram_config']['bot_token']
TELEGRAM_CHAT_ID = config['telegram_config']['chat_id']

MODEL_PATH_DETECTOR = '../models/deepfaune-yolov8s_960.pt'
MODEL_PATH_CLASSIFIER = '../models/deepfaune-vit_large_patch14_dinov2.lvd142m.pt'

ANIMAL_CLASSES = ["badger", "ibex", "red deer", "chamois", "cat", "goat", "roe deer", "dog", "squirrel", "equid", "genet",
                  "hedgehog", "lagomorph", "wolf", "lynx", "marmot", "micromammal", "mouflon",
                  "sheep", "mustelid", "bird", "bear", "nutria", "fox", "wild boar", "cow"]

class Detector:
    """
    Detector class to perform object detection using YOLO model.
    """
    def __init__(self):
        """
        Initialise the detector with a YOLO model.
        """
        self.model = YOLO(MODEL_PATH_DETECTOR)

    def bestBoxDetection(self, imagecv):
        """
        Detect the best bounding box in the given image.

        Args:
            imagecv (numpy.ndarray): Image in OpenCV format.

        Returns:
            tuple: Cropped image, class ID, bounding box coordinates, confidence score, and additional information (None).
        """
        image_rgb = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        resized_image = image_pil.resize((960, 960), Image.Resampling.LANCZOS)
        results = self.model(resized_image)

        if not results or not results[0].boxes or results[0].boxes.data.shape[0] == 0:
            return None, 0, np.zeros(4), 0, None

        detections = results[0].boxes.data
        best_detection = detections[detections[:, 4].argmax()]
        xmin, ymin, xmax, ymax, conf, cls_id = best_detection[:6]
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        cropped_image = resized_image.crop(box)
        return cropped_image, int(cls_id), box, conf, None

class Classifier:
    """
    Classifier class to classify detected objects using a ViT model.
    """
    def __init__(self):
        """
        Initialize the classifier with a ViT model and necessary transforms.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = timm.create_model('vit_large_patch14_dinov2', pretrained=False, num_classes=len(ANIMAL_CLASSES), dynamic_img_size=True)
        state_dict = torch.load(MODEL_PATH_CLASSIFIER, map_location=torch.device(device))['state_dict']
        self.model.load_state_dict({k.replace('base_model.', ''): v for k, v in state_dict.items()})
        self.transforms = transforms.Compose([
            transforms.Resize((182, 182), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model.eval()

    def predict(self, image):
        """
        Predict the class of the given image.

        Args:
            image (PIL.Image): Image to classify.

        Returns:
            tuple: Predicted animal type and confidence score.
        """
        img_tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            return ANIMAL_CLASSES[top_class.item()], top_p.item()

def process_single_image(image):
    """
    Process a single image to detect and classify animals.

    Args:
        image (PIL.Image): Image to process.

    Returns:
        tuple: Processed image and caption with detection details.
    """
    detector = Detector()
    classifier = Classifier()
    imagecv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cropped_image, cls_id, box, conf, _ = detector.bestBoxDetection(imagecv)
    if cropped_image is not None:
        animal_type, confidence = classifier.predict(cropped_image)
        caption = f"Detection: {animal_type} with {confidence * 100:.2f}% confidence"
        print(caption)
        return cropped_image, caption
    else:
        print("No animal of interest detected.")
        return None, None

def send_photo_to_telegram(bot_token, chat_id, photo, caption):
    """
    Send a photo to a specified Telegram chat.

    Args:
        bot_token (str): Telegram bot token.
        chat_id (str): Telegram chat ID.
        photo (PIL.Image): Photo to send.
        caption (str): Caption for the photo.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with io.BytesIO() as buf:
        photo.save(buf, format='JPEG')
        buf.seek(0)
        files = {'photo': buf}
        params = {'chat_id': chat_id, 'caption': caption}
        response = requests.post(url, files=files, data=params)
        response.raise_for_status()
        print("Alert sent.")

def download_image_from_url(url):
    """
    Download an image from a given URL.

    Args:
        url (str): URL of the image to download.

    Returns:
        PIL.Image: Downloaded image.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def extract_images_from_email(msg):
    """
    Extract images from an email message.

    Args:
        msg (email.message.Message): Email message to process.

    Returns:
        list: List of extracted images.
    """
    image_list = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get('Content-Disposition', '')

            if content_type.startswith('image/') and 'attachment' in content_disposition:
                print("Image received (Wilsus Camera).")
                image_data = part.get_payload(decode=True)
                image = Image.open(io.BytesIO(image_data))
                image_list.append(image)

            elif content_type == 'text/html':
                html_body = part.get_payload(decode=True).decode()
                # Extract URLs from <img> tags
                image_urls = re.findall(r'<img src="(https?://[^"]+)"', html_body)
                for url in image_urls:
                    print("Image received (UOVision Camera).")
                    image = download_image_from_url(url)
                    if image:
                        image_list.append(image)
    return image_list

def check_emails():
    """
    Check and process new emails for images.

    This function connects to the email server, fetches unseen emails,
    extracts images, processes each image, and sends the processed image
    to a specified Telegram chat.
    """
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select('inbox')
    typ, data = mail.search(None, 'UNSEEN')
    for num in data[0].split():
        typ, data = mail.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        images = extract_images_from_email(msg)
        total_images = len(images)
        for index, image in enumerate(images):
            print(f"Processing image {index + 1} of {total_images}...")
            processed_image, caption = process_single_image(image)
            if processed_image:
                send_photo_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, image, caption)
        print(f"\nMonitoring {EMAIL_USER} for new messages...")
    mail.logout()

if __name__ == "__main__":
    """
    Main loop to monitor email for new messages and process them.
    """
    print(f"Monitoring {EMAIL_USER} for new messages...")
    while True:
        try:
            time.sleep(1)
            check_emails()
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"\nMonitoring {EMAIL_USER} for new messages...")
            continue

