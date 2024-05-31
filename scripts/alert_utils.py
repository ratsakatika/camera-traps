#################################################
###### DETECTION AND CLASSIFICATION MODELS ######
#################################################

import torch
import timm
from datetime import datetime
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class classifier:
    """Image classifier for animal species."""
    def __init__(self, model_path_classifier, backbone, animal_classes, device='cpu'):
        self.model = timm.create_model(backbone, pretrained=False, num_classes=len(animal_classes))
        state_dict = torch.load(model_path_classifier, map_location=torch.device(device))['state_dict']
        self.model.load_state_dict({k.replace('base_model.', ''): v for k, v in state_dict.items()})
        self.transforms = transforms.Compose([
            transforms.Resize((182, 182), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model.eval()
        self.animal_classes = animal_classes

    def predict(self, image):
        """Predict the species of an animal in the image."""
        img_tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            return self.animal_classes[top_class.item()], top_p.item()

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


#################################################
######## CHECK EMAILS AND EXTRACT DATA ##########
#################################################

import imaplib
import email
import io
import re
import requests
from PIL import Image
from email.utils import parsedate_to_datetime
from PIL.ExifTags import TAGS

def check_emails(IMAP_HOST, EMAIL_USER, EMAIL_PASS):
    """Check emails for new messages with images and extract metadata."""
    images = []
    camera_id = None
    temp_deg_c = None
    img_date = None
    img_time = None
    battery = None
    sd_memory = None
    
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select('inbox')
    typ, data = mail.search(None, 'UNSEEN')

    if data[0].split():

        oldest_unread = data[0].split()[0]
        typ, data = mail.fetch(oldest_unread, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])

        images = extract_images_from_email(msg)
        subject = msg['subject']
        body = get_email_body(msg)
        camera_id = extract_camera_id(subject, body)
        temp_deg_c = extract_temperature(body)
        if images:
            img_date, img_time = extract_date_time_from_image(images[0])
        if img_date == None:
            img_date = extract_date(body)
        if img_time == None:
            img_time = extract_time(body)
        battery = extract_battery(body)
        sd_memory = extract_sd_free_space(body)
        
    mail.logout()

    return images, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory

def get_email_body(msg):
    """Extract the body from an email message."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain":
                body = part.get_payload(decode=True).decode()
                break
            elif content_type == "text/html":
                body = part.get_payload(decode=True).decode()

    else:
        body = msg.get_payload(decode=True).decode()

    return body


def extract_images_from_email(msg):
    """Extract images from an email message."""
    image_list = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get('Content-Disposition', '')

            if content_type.startswith('image/') and 'attachment' in content_disposition:
                image_data = part.get_payload(decode=True)
                image = Image.open(io.BytesIO(image_data))
                image_list.append(image)
            elif content_type == 'text/html':
                html_body = part.get_payload(decode=True).decode()
                image_urls = re.findall(r'<img src="(https?://[^"]+)"', html_body)
                for url in image_urls:
                    image = download_image_from_url(url)
                    if image:
                        image_list.append(image)
    return image_list

def download_image_from_url(url):
    """Download an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def extract_camera_id(subject, body):
    # Extract camera ID from subject
    camera_id = None
    if subject:
        camera_id = subject.split('_')[-1]

    # Extract camera ID from body
    camera_id_match = re.search(r'Camera ID:\s*(\w+)|CamID:\s*(\w+)', body)
    if camera_id_match:
        camera_id = camera_id_match.group(1) or camera_id_match.group(2)
    
    return camera_id

def extract_date_time_from_image(image):
    try:
        exif_data = image._getexif()
        if exif_data:
            date_time_original = exif_data.get(36867)
            if date_time_original:
                date_str, time_str = date_time_original.split(' ')
                date_obj = datetime.strptime(date_str, '%Y:%m:%d')
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                return date_obj.strftime('%Y-%m-%d'), time_obj.strftime('%H:%M:%S')
    except Exception as e:
        print(f"Error extracting date and time from image: {e}")
    return None, None

def extract_date(body):
    date_patterns = [
        r"Date:(\d{2}\.\d{2}\.\d{4})",      # Pattern for "Date:31.05.2024"
        r"Date:(\d{2}/\d{2}/\d{4})",        # Pattern for "Date:30/05/2024"
        r"Date & Time:\((\d{2}/\d{2}/\d{4})" # Pattern for "Date & Time:(31/05/2024"
    ]
    for pattern in date_patterns:
        match = re.search(pattern, body)
        if match:
            date_str = match.group(1)
            try:
                if '.' in date_str:
                    date_obj = datetime.strptime(date_str, '%d.%m.%Y')
                elif '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
    return None

def extract_time(body):
    time_patterns = [
        r"Time:(\d{2}:\d{2}:\d{2})",        # Pattern for "Time:17:08:02"
        r"Date:(?:\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})", # Pattern for "Date:30/05/2024  17:02:06"
        r"Date & Time:\((?:\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})" # Pattern for "Date & Time:(31/05/2024  17:02:20)"
    ]
    for pattern in time_patterns:
        match = re.search(pattern, body)
        if match:
            return match.group(1)
    return None

def extract_temperature(body):
    temp_patterns = [
        r"Temperature:(\d+)℃",             # Pattern for "Temperature:21℃"
        r"Temp:(\d+)\s*Celsius\s*Degree",  # Pattern for "Temp:19 Celsius Degree"
        r"Temp:(\d+)\s*C"                  # Pattern for "Temp:19C" (if exists in future emails)
    ]
    
    for pattern in temp_patterns:
        match = re.search(pattern, body)
        if match:
            return int(match.group(1))
    return None

def extract_sd_free_space(body):
    sd_patterns = [
        r"SD card free space:\s*([\d\.]+)GB of ([\d\.]+)GB",  # Pattern for "SD card free space: 14.7GB of 14.8GB"
        r"SD:(\d+)M/(\d+)M"                                  # Pattern for "SD:15115M/15189M"
    ]
    
    for pattern in sd_patterns:
        match = re.search(pattern, body)
        if match:
            free_space = float(match.group(1))
            total_space = float(match.group(2))
            return int((free_space / total_space) * 100)
    return None

def extract_battery(body):
    battery_patterns = [
        r"Battery:(\d+)%",  # Pattern for "Battery:100%"
    ]
    
    for pattern in battery_patterns:
        match = re.search(pattern, body)
        if match:
            return int(match.group(1))
    return None


##############################################################################
######## EXTRACT CAMERA LOCATIONS AND UPDATE BATTERY/STORAGE STATUS ##########
##############################################################################

import pandas as pd

def extract_update_camera_status(camera_id, battery=None, sd_memory=None):
    # Read the CSV file
    df = pd.read_csv('../data/camera_locations.csv')

    # Find the row with the matching Camera ID
    camera_row = df[df['Camera ID'] == camera_id]

    if camera_row.empty:
        raise ValueError(f"No camera found with ID {camera_id}")

    # Extract the details
    camera_make = camera_row['Make'].values[0]
    gps = f"X: {camera_row['X'].values[0]}, Y: {camera_row['Y'].values[0]}, Z: {camera_row['Z'].values[0]}m"
    location = camera_row['Toponym'].values[0]
    map_url = camera_row['Google Link'].values[0]

    # Get or update the battery and SD memory values
    if battery is None:
        battery = camera_row['Battery %'].values[0]
    else:
        df.loc[df['Camera ID'] == camera_id, 'Battery %'] = battery

    if sd_memory is None:
        sd_memory = camera_row['SD Memory %'].values[0]
    else:
        df.loc[df['Camera ID'] == camera_id, 'SD Memory %'] = sd_memory

    # Save the updated CSV file
    df.to_csv('../data/camera_locations.csv', index=False)

    return camera_make, gps, location, map_url, battery, sd_memory