#################################################
############ IMAGE PROCESSING TOOLS #############
#################################################

import torch
import timm
from datetime import datetime
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import io
from PIL import Image
import ast
from IPython.display import display


from collections import Counter

def set_device():
    """
    Determine and return the best device to run the model on.

    Returns:
        - (str): 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def image_to_bytes(image):
    """
    Convert an image to bytes.

    Parameters:
        - image: The input image to be converted to bytes.

    Returns:
        - (BytesIO): The image converted to bytes in JPEG format.
    """
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr


from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
import torch
import os
# import tempfile
import numpy as np


def detector(df, model, images, DETECTION_THRESHOLD):
    """
    Runs MegaDetector on a list of images and updates a dataframe with detection results.

    Parameters:
        - df (dataframe): The dataframe to be updated with detection results.
        - model (object): The MegaDetector model used for generating detections.
        - images (list): A list of images to be processed.
        - DETECTION_THRESHOLD (float): The confidence threshold above which detections are considered valid.

    Returns:
        - df (dataframe): The updated dataframe with detection results, including detection boxes, classes, confidences, and counts of detected animals, humans, and vehicles.
        - human_warning (bool): A flag indicating whether any humans or vehicles were detected (True if detected, False otherwise).

    Raises:
        - ValueError: If the dataframe does not have enough rows to update for the given number of images.
    """

    human_warning = False
    num_images = len(images)
    
    # Ensure there are enough rows in the DataFrame to update
    if len(df) < num_images:
        raise ValueError(f"{current_time()} | Critical Error: The DataFrame does not have enough rows to update.")
    
    for i, image in enumerate(images):

        processed_image = np.array(image)
        result = model.single_image_detection(processed_image)

        detections_above_threshold = []
        for idx, conf in enumerate(result['detections'].confidence):
            if conf > DETECTION_THRESHOLD:
                detections_above_threshold.append({
                    'bbox': result['detections'].xyxy[idx],  # Bounding box
                    'category': result['detections'].class_id[idx],  # Class ID
                    'conf': conf  # Confidence
                })

        detection_boxes = [d['bbox'].tolist() for d in detections_above_threshold]
        detection_classes = [int(d['category']) for d in detections_above_threshold] # +1 to convert back to original class IDs
        detection_confidences = [float(d['conf']) for d in detections_above_threshold]

        animal_count = sum(1 for d in detections_above_threshold if d['category'] == 0)
        human_count = sum(1 for d in detections_above_threshold if d['category'] == 1)
        vehicle_count = sum(1 for d in detections_above_threshold if d['category'] == 2)

        print(f"{current_time()} | Image {i+1}: Animal Count = {animal_count}, Human Count = {human_count}, Vehicle Count = {vehicle_count}")

        # Update the respective row in the DataFrame
        df.at[df.index[-num_images + i], 'Detection Boxes'] = detection_boxes
        df.at[df.index[-num_images + i], 'Detection Classes'] = detection_classes
        df.at[df.index[-num_images + i], 'Detection Confidences'] = detection_confidences
        df.at[df.index[-num_images + i], 'Animal Count'] = animal_count
        df.at[df.index[-num_images + i], 'Human Count'] = human_count
        df.at[df.index[-num_images + i], 'Vehicle Count'] = vehicle_count

        # Set the human_warning trigger to true if humans or vehicles are detected
        if human_count > 0 or vehicle_count > 0:
            human_warning = True
            

    return df, human_warning


class classifier:
    """
    A classifier for predicting the species of animals in images.

    Parameters:
        - CLASSIFIER_MODEL_PATH (str): The path to the pre-trained classifier model file.
        - backbone (str): The name of the model architecture to be used as the backbone.
        - animal_classes (list): A list of animal class names that the model can predict.
        - device (str, optional): The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.

    Methods:
        - predict(image): Predict the species of an animal in the given image.

    Example:
        classifier_instance = classifier(CLASSIFIER_MODEL_PATH, BACKBONE, animal_classes, device='cuda')
        prediction, probability = classifier_instance.predict(image)
    """
    def __init__(self, CLASSIFIER_MODEL_PATH, backbone, animal_classes, device='cpu'):
        self.model = timm.create_model(backbone, pretrained=False, num_classes=len(animal_classes), dynamic_img_size=True)

        state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)

        # Handle the case where the state_dict key exists
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove 'base_model.' prefix if it exists
        if any(k.startswith('base_model.') for k in state_dict.keys()):
            state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items()}
        
        # Remove 'model.' prefix if it exists
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        print(f"{current_time()} | Model state dictionary loaded successfully.")

        self.transforms = transforms.Compose([
            transforms.Resize((182, 182), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model.eval()
        self.animal_classes = animal_classes

    def predict(self, image):
        """
        Predict the species of an animal in the image.

        Parameters:
            - image: The input image to be classified.

        Returns:
            - (str): The predicted animal class.
            - (float): The confidence of the prediction.
        """
        img_tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            return self.animal_classes[top_class.item()], top_p.item()


def batch_classification(df, classifier_model, images, CLASSIFICATION_THRESHOLD=0.3, PRIORITY_SPECIES=[""]):
    """
    Performs batch classification on a list of images and update the dataframe with species information.

    Parameters:
        - df (dataframe): The dataframe to be updated with classification results.
        - classifier_model (object): The classifier model used for predicting species.
        - images (list): A list of images to be processed.

    Returns:
        - df (dataframe): The updated dataframe with classification results, including species classes, confidences, primary species, and counts of wild boar and bear.
    """
    num_images = len(images)
    df_length = len(df)
    priority_detections = []

    for i, image in enumerate(images):

        species_list = []
        species_confidence_list = []
        primary_species_list = []
        detection_boxes = df['Detection Boxes'][df_length-num_images + i]

        if detection_boxes == []:

            primary_species_list.append("Empty")
            print(f"{current_time()} | Image {i+1}, No Detections")

        else:

            detection_classes = df['Detection Classes'][df_length-num_images + i]
            detection_conf = df['Detection Confidences'][df_length-num_images + i]

            for j, bbox in enumerate(detection_boxes):
                
                if detection_classes[j] == 0:  # Only classify if an animal

                    left, top, right, bottom = bbox  # Unpack the bounding box

                    cropped_image = image.crop((left, top, right, bottom))

                    species, species_conf = classifier_model.predict(cropped_image)

                    species_list.append(species)
                    species_confidence_list.append(species_conf)

                    if species in PRIORITY_SPECIES and species_conf >= CLASSIFICATION_THRESHOLD:
                        priority_detections.append(species)

                    print(f"{current_time()} | Image {i+1}, Detection {j+1} ({detection_conf[j] * 100:.2f}% confidence), Species: {species} ({species_conf * 100:.2f}% confidence)")
                    

        # Check for valid animal detections above the threshold
        filtered_species = [species for species, confidence in zip(species_list, species_confidence_list) if confidence >= CLASSIFICATION_THRESHOLD]

        if filtered_species:  # If there are valid species detections
            # Count the occurrences of each species
            species_counter = Counter(filtered_species)
            most_common_count = max(species_counter.values())

            # Identify species with the highest count
            most_common_species = [species for species, count in species_counter.items() if count == most_common_count]

            # If there's a tie, select the species with the highest confidence
            if len(most_common_species) > 1:
                max_confidence = -1
                max_species_confidence_name = None

                for species in most_common_species:
                    indices = [i for i, x in enumerate(species_list) if x == species]
                    max_species_confidence = max(species_confidence_list[i] for i in indices)
                    if max_species_confidence > max_confidence:
                        max_confidence = max_species_confidence
                        max_species_confidence_name = species

                primary_species_list.append(max_species_confidence_name)
            else:
                primary_species_list.append(most_common_species[0])

        else:  # If no valid animal detections
            # Check for humans or vehicles
            if df['Human Count'][df_length-num_images + i] > 0:
                print(f"{current_time()} | Image {i+1}, No Animals Detected - Humans/Vehicles Only")
                primary_species_list.append("Human")  # If humans are detected

            elif df['Vehicle Count'][df_length-num_images + i] > 0:
                print(f"{current_time()} | Image {i+1}, No Animals Detected - Humans/Vehicles Only")
                primary_species_list.append("Vehicle")  # If vehicles are detected

            else:
                print(f"{current_time()} | Image {i+1}, No Animals/Humans/Vehicles Detected")
                primary_species_list.append("Empty")  # If no animals, humans, or vehicles are detected



        df.at[df.index[-num_images + i], 'Species Classes'] = species_list
        df.at[df.index[-num_images + i], 'Species Confidences'] = species_confidence_list
        df.at[df.index[-num_images + i], 'Primary Species'] = primary_species_list[0]

    # Remove duplicates from the priority species detection list
    priority_detections = list(set(priority_detections))

    return df, priority_detections

def detections_in_sequence(df, images, CLASSIFICATION_THRESHOLD):
    """
    Checks if there are any animal detections > CLASSIFICATION_THRESHOLD or human/vehicle detections of any confidence level

    Parameters:
        - df (dataframe): The dataframe containing detection data.
        - images (list): A list of images in the sequence.
        - 

    Returns:
        - (bool): True if any human or vehicle detections are present, or if the primary species is not empty. False otherwise.
    """

    last_rows = df.iloc[-len(images):]

    # Flatten the Species Confidences column for the selected rows
    all_confidences = [conf for sublist in last_rows['Species Confidences'] for conf in sublist]

    # Check if any of the confidences are greater than CLASSIFICATION_THRESHOLD
    confidences_above_threshold = any(conf > CLASSIFICATION_THRESHOLD for conf in all_confidences)

    # Check if humans or vehicles are present in any of the photos
    human_or_vehicle_present = (last_rows['Human Count'] > 0).any() or (last_rows['Vehicle Count'] > 0).any()

    return human_or_vehicle_present or confidences_above_threshold

##################################################################
######## CHECK EMAILS, EXTRACT DATA, SEND WEEKLY REPORT ##########
##################################################################

import imaplib
import email
import io
import re
import requests
from PIL import Image
from email.utils import parsedate_to_datetime
from PIL.ExifTags import TAGS

import imaplib
import email
from email.header import decode_header

def check_emails(IMAP_HOST, EMAIL_USER, EMAIL_PASS):
    """
    Check emails for new messages with images and extract metadata.

    Parameters:
        - IMAP_HOST (str): The IMAP host address for the email server.
        - EMAIL_USER (str): The email address to log in.
        - EMAIL_PASS (str): The password for the email address.

    Returns:
        - images (list): A list of extracted images.
        - original_images (list): A list of copies of the original images.
        - camera_id (str): The ID of the camera extracted from the email.
        - temp_deg_c (float): The temperature in degrees Celsius extracted from the email.
        - img_date (str): The date extracted from the image or email.
        - img_time (str): The time extracted from the image or email.
        - battery (str): The battery level extracted from the email.
        - sd_memory (str): The free space on the SD card extracted from the email.
    """
    # Initialize variables to store email content and metadata
    images = []  # List to store extracted images
    image_count = None # Stores the number of images
    original_images = []  # List to store copies of the original images
    camera_id = None  # Camera ID will be extracted from email
    temp_deg_c = None  # Temperature will be extracted from email
    img_date = None  # Image date will be extracted from image or email
    img_time = None  # Image time will be extracted from image or email
    battery = None  # Battery level will be extracted from email
    sd_memory = None  # Free SD card space will be extracted from email

    # Connect to the email server using IMAP
    mail = imaplib.IMAP4_SSL(IMAP_HOST)  # Connect to the email server securely
    mail.login(EMAIL_USER, EMAIL_PASS)  # Login to the email account with the provided credentials
    mail.select('inbox')  # Select the inbox folder to search for emails

    # Search for unread (UNSEEN) messages in the inbox
    typ, data = mail.search(None, 'UNSEEN')  # Search for unread emails

    # If there are any unread emails, process the first one
    if data[0].split():  # Check if there are any unread emails

        print(f"{current_time()} | *** Email Received ***")

        # Get the ID of the oldest unread email
        oldest_unread = data[0].split()[0]  # Get the ID of the first unread email
        typ, data = mail.fetch(oldest_unread, '(RFC822)')  # Fetch the email content
        msg = email.message_from_bytes(data[0][1])  # Parse the email content from bytes to message format

        # Extract the subject and body of the email
        subject = msg['subject']  # Get the subject of the email
        body = get_email_body(msg)  # Get the body of the email, including any plain text or HTML content
        
        # Extract metadata from the email content
        camera_id = extract_camera_id(subject, body)  # Extract camera ID from the subject or body
        temp_deg_c = extract_temperature(body)  # Extract the temperature from the email body
        battery = extract_battery(body)  # Extract battery level from the body
        sd_memory = extract_sd_free_space(body)  # Extract SD card free space from the body

        # Extract images from the email's attachments
        images = extract_images_from_email(msg)  # This function will extract any image attachments
        original_images = [img.copy() for img in images]  # Create copies of the images for safe keeping

        # Count the number of images received
        image_count = len(images)

        # If images are available, extract the date and time from the first image
        if images:
            img_date, img_time = extract_date_time_from_image(images[0])  # Extract date and time from the first image's metadata
        if img_date == None:
            img_date = extract_date(body)  # Extract the date from the body if not found in the image
        if img_time == None:
            img_time = extract_time(body)  # Extract the time from the body if not found in the image
        
    # Log out of the email server
    mail.logout()  # Log out from the IMAP server after processing the email

    # Return the extracted data
    return images, original_images, image_count, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory



def get_email_body(msg):
    """
    Extract the body from an email message.

    Parameters:
        - msg (email.message.EmailMessage): The email message object.

    Returns:
        - body (str): The extracted body of the email as a string.
    """
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
    """
    Extract images from an email message, handling multiple content types.

    Parameters:
        - msg (email.message.EmailMessage): The email message object.

    Returns:
        - image_list (list): A list of extracted images.
    """

    def download_image_from_url(url):
        """Download image from URL and return as PIL Image."""
        import requests
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
            return None

    def is_valid_image_filename(filename):
        """Check if a given filename has a valid image extension."""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        return filename.lower().endswith(valid_extensions)


    # print("Processing email for images...")
    image_list = []

    if not msg.is_multipart():
        # print("Email is not multipart; skipping.")
        return image_list

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = part.get('Content-Disposition', '')

        # print(f"Processing part: {content_type}, disposition: {content_disposition}")

        try:
            # Handle known image content types
            if content_type.startswith('image/'):
                # print(f"Extracting image from: {content_type}")
                image_data = part.get_payload(decode=True)
                image = Image.open(io.BytesIO(image_data))
                image_list.append(image)

            # Handle generic binary attachments that might be images
            elif content_type in ['application/octet-stream', 'application/x-binary']:
                filename = part.get_filename()
                if filename and is_valid_image_filename(filename):
                    # print(f"Extracting image from binary attachment: {filename}")
                    image_data = part.get_payload(decode=True)
                    image = Image.open(io.BytesIO(image_data))
                    image_list.append(image)

            # Handle embedded images (cid:)
            elif content_type == 'text/html':
                html_body = part.get_payload(decode=True).decode(errors='ignore')
                # Extract inline images with Content-ID references
                inline_images = re.findall(r'<img src="cid:([^"]+)"', html_body)
                for cid in inline_images:
                    cid_part = next((p for p in msg.walk() if p.get('Content-ID') == f"<{cid}>"), None)
                    if cid_part:
                        image_data = cid_part.get_payload(decode=True)
                        image = Image.open(io.BytesIO(image_data))
                        image_list.append(image)
                        print(f"Extracted inline image with CID: {cid}")

                # Extract externally linked images
                external_images = re.findall(r'<img src="(https?://[^"]+)"', html_body)
                for url in external_images:
                    image = download_image_from_url(url)
                    if image:
                        image_list.append(image)

        except Exception as e:
            print(f"Error processing email part: {e}")

    if not image_list:
        print("No images found in the email.")

    return image_list



def download_image_from_url(url):
    """
    Download an image from a URL.

    Parameters:
        - url (str): The URL of the image to download.

    Returns:
        - (Image or None): The downloaded image if successful, otherwise None.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        print(f"{current_time()} | Error downloading image from {url}: {str(e)}")
        return None


def extract_camera_id(subject, body):
    """
    Extract the camera ID from the email subject or body.

    Parameters:
        - subject (str): The subject of the email.
        - body (str): The body of the email.

    Returns:
        - camera_id (str): The extracted camera ID.
    """

    # Extract camera ID from body (ID located in body of email - Wilsus)
    camera_id_match = re.search(r'Camera ID:\s*(\w+)|CamID:\s*(\w+)', body)
    if camera_id_match:
        camera_id = camera_id_match.group(1) or camera_id_match.group(2)
    
    # Otherwise, extract from the end of the subject line (UOVISION camera)
    elif subject and '_' in subject:
        camera_id = subject.split('_')[-1]
    
    # Otherwise return "Unknown"
    else:
        camera_id = "Unknown"
    
    return camera_id


def extract_date_time_from_image(image):
    """
    Extract the date and time from the EXIF data of an image.

    Parameters:
        - image (PIL.Image): The image from which to extract the date and time.

    Returns:
        - (tuple): A tuple containing:
            - date (str): The extracted date in 'YYYY-MM-DD' format, or None if extraction fails.
            - time (str): The extracted time in 'HH:MM:SS' format, or None if extraction fails.
    """
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
        print(f"{current_time()} | Error extracting date and time from image: {e}")
    return None, None


def extract_date(body):
    """
    Extract the date from the email body using predefined patterns.

    Parameters:
        - body (str): The body of the email.

    Returns:
        - date (str or None): The extracted date in 'YYYY-MM-DD' format, or None if no valid date is found.
    """
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
    """
    Extract the time from the email body using patterns found in UOVISION and WILSUS cameras.

    Parameters:
        - body (str): The body of the email.

    Returns:
        - time (str or None): The extracted time in 'HH:MM:SS' format, or None if no valid time is found.
    """
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
    """
    Extract the temperature from the email body using UOVISION and WILSUS patterns.

    Parameters:
        - body (str): The body of the email.

    Returns:
        - temperature (int or None): The extracted temperature in degrees Celsius, or None if no valid temperature is found.
    """
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
    """
    Extract the SD card free space percentage from the email body using UOVISION and WILSUS patterns.

    Parameters:
        - body (str): The body of the email.

    Returns:
        - sd_free_space (int or None): The percentage of free space on the SD card, or None if no valid information is found.
    """
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
    """
    Extract the battery level from the email body using UOVISION and WILSUS patterns.

    Parameters:
        - body (str): The body of the email.

    Returns:
        - battery_level (int or None): The extracted battery level as a percentage, or None if no valid information is found.
    """
    battery_patterns = [
        r"Battery:(\d+)%",  # Pattern for "Battery:100%"
    ]
    
    for pattern in battery_patterns:
        match = re.search(pattern, body)
        if match:
            return int(match.group(1))
    return None


import smtplib
import os
import zipfile
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime
import pandas as pd
import time

import zipfile
import os

def send_weekly_report_original(SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, SMTP_PORT, CAPTURE_DATABASE_PATH, CAMERA_LOCATIONS_PATH, RECIPIENTS, EMAIL_USER):
    """
    Send a weekly report email with attached zip file containing the CSVs.

    Parameters:
        - SMTP_SERVER (str): The SMTP server address.
        - EMAIL_SENDER (str): The sender's email address.
        - EMAIL_PASSWORD (str): The password for the sender's email account.
        - SMTP_PORT (int): The port number for the SMTP server.
        - CAPTURE_DATABASE_PATH (str): The path to the capture database file to be attached.
        - CAMERA_LOCATIONS_PATH (str): The path to the camera locations file to be attached.
        - RECIPIENTS (list): A list of recipient email addresses.
        - EMAIL_USER (str): The email address used for monitoring.

    Returns:
        - None
    """
    
    # Create the zip file
    zip_filename = "weekly_report.zip"
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(CAPTURE_DATABASE_PATH, os.path.basename(CAPTURE_DATABASE_PATH))
            zipf.write(CAMERA_LOCATIONS_PATH, os.path.basename(CAMERA_LOCATIONS_PATH))
        print(f"\n{current_time()} | Created zip file: {zip_filename}")
    except Exception as e:
        print(f"\n{current_time()} | Error creating zip file: {e}")
        return  # Exit the function if zipping fails

    subject = "Camera Trap Alert System: Weekly Report"
    body = "Good morning,\n\nPlease see attached the latest versions of:\n\n(1) The alert system capture database.\n(2) The camera trap status log.\n\nWeekly reports are sent every Monday at 08:00.\n\nBest regards,\n\The Carpathia Foundation Wildlife Alert System"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(RECIPIENTS)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach the zip file
    try:
        with open(zip_filename, 'rb') as zip_file:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(zip_file.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename={zip_filename}')
            msg.attach(attachment)
        print(f"\n{current_time()} | Zip file attached successfully.")
    except Exception as e:
        print(f"\n{current_time()} | Error attaching zip file to weekly report: {e}")
        return

    # Send the email with the zip file attached
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"\n{current_time()} | Weekly report email sent successfully.")
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
    except Exception as e:
        print(f"\n{current_time()} | Failed to send weekly report email: {e}")
        print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")



import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime
import pandas as pd
import time

def send_weekly_report(SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, SMTP_PORT, CAPTURE_DATABASE_PATH, CAMERA_LOCATIONS_PATH, RECIPIENTS, EMAIL_USER):
    subject = "Camera Trap Alert System: Weekly Report"
    body = "Good morning,\n\nPlease see attached the latest versions of:\n\n(1) The alert system capture database.\n(2) The camera trap status log.\n\nWeekly reports are sent every Monday at 08:00.\n\nBest regards,\n\nFCC Camera Trap Automatic Alert System"

    attachments = [CAPTURE_DATABASE_PATH, CAMERA_LOCATIONS_PATH]

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(RECIPIENTS)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    for file_path in attachments:
        attachment = MIMEBase('application', 'octet-stream')
        try:
            with open(file_path, 'rb') as file:
                attachment.set_payload(file.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
            msg.attach(attachment)
        except Exception as e:
            print(f"\n{current_time()} | Error attaching file to weekly report {file_path}: {e}")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"\n{current_time()} | Weekly report email sent successfully.")
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
    except Exception as e:
        print(f"\n{current_time()} | Failed to send weekly report email: {e}")
        print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")






def send_weekly_report_new(SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, SMTP_PORT, CAPTURE_DATABASE_PATH, CAMERA_LOCATIONS_PATH, RECIPIENTS, EMAIL_USER):
    """
    Send a weekly report email with attached files.
    """
    subject = "Weekly Wildlife Monitoring Report"
    body = f"<b>Weekly Wildlife Monitoring Report:</b> {current_time()}<br><br>"

    # Load the CSVs into DataFrames
    df_capture = pd.read_csv(CAPTURE_DATABASE_PATH)
    df_camera = pd.read_csv(CAMERA_LOCATIONS_PATH)
    
    # Get the current date
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(days=7)

    # Filter the data for the last 7 days (including yesterday)
    df_capture_last_week = df_capture[(pd.to_datetime(df_capture['Date']) >= start_date) & (pd.to_datetime(df_capture['Date']) <= end_date)]

    # Group by sequence ID and calculate the max animal count for each sequence
    df_capture_last_week['Max Animal Count'] = df_capture_last_week.groupby('Sequence ID')['Animal Count'].transform('max')

    # Create a table of species sightings
    species_count = df_capture_last_week.groupby('Primary Species')['Max Animal Count'].sum().reset_index()
    species_count = species_count.sort_values(by='Max Animal Count', ascending=False)

    # Calculate weekly differences for species
    species_count['Difference vs. Previous Week'] = species_count['Max Animal Count'] - species_count['Max Animal Count'].shift(1, fill_value=0)
    species_count['Difference vs. Weekly Average'] = species_count['Max Animal Count'] - species_count['Max Animal Count'].mean()

    # Build the Table 1 string
    table_1 = "<b>Table 1: Wildlife Capture Statistics</b> ({start_date.strftime('%d-%m')} - {end_date.strftime('%d-%m')})<br>"
    table_1 += "<table border='1' style='border-collapse: collapse;'><tr><th>Species</th><th># Sightings</th><th>Difference vs. Previous Week</th><th>Difference vs. Weekly Average</th></tr>"
    for _, row in species_count.iterrows():
        table_1 += f"<tr><td>{row['Primary Species']}</td><td>{row['Max Animal Count']}</td><td>{row['Difference vs. Previous Week']}</td><td>{row['Difference vs. Weekly Average']}</td></tr>"
    table_1 += "</table><br>"

    # Build the Table 2 string
    # Sort df_camera by 'Last Communication' to ensure cameras are ordered by most recent communication first
    df_camera_sorted = df_camera.sort_values(by='Last Updated', ascending=False)

    table_2 = "<b>Table 2: Camera Trap Statistics</b><br>"
    table_2 += "<table border='1' style='border-collapse: collapse;'><tr><th>Camera ID</th><th>Location</th><th>Images Sent</th><th>Detection Events</th><th>Animals</th><th>Humans</th><th>Vehicles</th><th>Battery %</th><th>SD %</th><th>Last Communication</th></tr>"

    camera_ids = df_capture_last_week['Camera ID'].unique()

    # Track cameras not found in the camera_locations CSV (for Unknown category)
    unknown_cameras = df_capture_last_week[~df_capture_last_week['Camera ID'].isin(df_camera['Camera ID'])]

    # Merge the camera stats with the camera info
    for _, row in df_camera_sorted.iterrows():
        camera_id = row['Camera ID']
        location = row['Toponym']
        images_sent = len(df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id])
        detection_events = len(df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id].drop_duplicates(subset=['Sequence ID']))
        animals = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Max Animal Count'].sum()
        humans = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Human Count'].sum()
        vehicles = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Vehicle Count'].sum()
        battery = row['Battery %']
        sd_memory = row['SD Memory %']
        last_communication = row['Last Updated']

        table_2 += f"<tr><td>{camera_id}</td><td>{location}</td><td>{images_sent}</td><td>{detection_events}</td><td>{animals}</td><td>{humans}</td><td>{vehicles}</td><td>{battery}</td><td>{sd_memory}</td><td>{last_communication}</td></tr>"

    # Add the Unknown category for cameras that aren't in the camera_locations CSV
    for camera_id in unknown_cameras['Camera ID'].unique():
        location = "Unknown"
        images_sent = len(df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id])
        detection_events = len(df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id].drop_duplicates(subset=['Sequence ID']))
        animals = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Max Animal Count'].sum()
        humans = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Human Count'].sum()
        vehicles = df_capture_last_week[df_capture_last_week['Camera ID'] == camera_id]['Vehicle Count'].sum()
        battery = "N/A"
        sd_memory = "N/A"
        last_communication = "N/A"

        table_2 += f"<tr><td>{camera_id}</td><td>{location}</td><td>{images_sent}</td><td>{detection_events}</td><td>{animals}</td><td>{humans}</td><td>{vehicles}</td><td>{battery}</td><td>{sd_memory}</td><td>{last_communication}</td></tr>"

    table_2 += "</table><br>"

    # Prepare the email body
    body += f"During the week ending {end_date.strftime('%d %b %Y')}, the wildlife alert system processed {len(df_capture_last_week)} images from {len(df_capture_last_week['Sequence ID'].unique())} detection events.<br><br>"
    body += table_1 + "<br><br>" + table_2 + "<br><br>"

    # Send the email with the attached report
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(RECIPIENTS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    # Attach the CSVs as zipped files
    with zipfile.ZipFile('/tmp/reports.zip', 'w') as zipf:
        zipf.write(CAPTURE_DATABASE_PATH, os.path.basename(CAPTURE_DATABASE_PATH))
        zipf.write(CAMERA_LOCATIONS_PATH, os.path.basename(CAMERA_LOCATIONS_PATH))

    with open('/tmp/reports.zip', 'rb') as f:
        attach = MIMEBase('application', 'octet-stream')
        attach.set_payload(f.read())
        encoders.encode_base64(attach)
        attach.add_header('Content-Disposition', 'attachment', filename="reports.zip")
        msg.attach(attach)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"\n{current_time()} | Weekly report email sent successfully.")
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
    except Exception as e:
        print(f"\n{current_time()} | Failed to send weekly report email: {e}")
        print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")





##############################################################################
######## EXTRACT CAMERA LOCATIONS AND UPDATE BATTERY/STORAGE STATUS ##########
##############################################################################


import pandas as pd

def extract_and_update_camera_info(CAMERA_LOCATIONS_PATH, camera_id, battery=None, sd_memory=None):
    """
    Extract and update camera information based on the given camera ID.

    Parameters:
        - CAMERA_LOCATIONS_PATH (str): The path to the CSV file containing camera information.
        - camera_id (str): The ID of the camera to be looked up.
        - battery (int, optional): The battery level to update, if provided.
        - sd_memory (int, optional): The SD memory percentage to update, if provided.

    Returns:
        - camera_make (str or None): The make of the camera, or None if no camera is found.
        - gps (str or None): The GPS coordinates of the camera, or None if no camera is found.
        - location (str or None): The location of the camera, or None if no camera is found.
        - map_url (str or None): The Google Maps link of the camera location, or None if no camera is found.
        - battery (int or None): The battery level of the camera, or None if no camera is found or updated.
        - sd_memory (int or None): The SD memory percentage of the camera, or None if no camera is found or updated.
    """

    # Read the CSV file
    df = pd.read_csv(CAMERA_LOCATIONS_PATH)

    # Find the row with the matching Camera ID
    camera_row = df[df['Camera ID'] == camera_id]

    if camera_row.empty:
        print(f"{current_time()} | Camera ID '{camera_id}' not found in camera database. Please update: {CAMERA_LOCATIONS_PATH}")
        return None, None, None, None, battery, None

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

    df.loc[df['Camera ID'] == camera_id, 'Last Updated'] = current_time()

    # Save the updated CSV file
    df.to_csv('../data/camera_locations.csv', index=False)

    return camera_make, gps, location, map_url, battery, sd_memory


def update_camera_data_dataframe(df, images_count, camera_id, camera_make, img_date, img_time, temp_deg_c, battery, sd_memory, location, gps, map_url):
    """
    Update the camera data dataframe with new image metadata and camera information.

    Parameters:
        - df (dataframe): The existing dataframe to update.
        - images_count (int): The number of images being added.
        - camera_id (str): The ID of the camera.
        - camera_make (str): The make of the camera.
        - img_date (str): The date the images were taken.
        - img_time (str): The time the images were taken.
        - temp_deg_c (float or str): The temperature in degrees Celsius.
        - battery (int or str): The battery level of the camera.
        - sd_memory (int or str): The SD memory percentage of the camera.
        - location (str): The location of the camera.
        - gps (str): The GPS coordinates of the camera.
        - map_url (str): The Google Maps link of the camera location.

    Returns:
        - df (dataframe): The updated dataframe with the new image metadata and camera information.
    """

    camera_id = "" if camera_id is None else camera_id
    camera_make = "" if camera_make is None else camera_make
    location = "" if location is None else location
    gps = "" if gps is None else gps
    battery = "" if battery is None else battery
    sd_memory = "" if sd_memory is None else sd_memory
    map_url = "" if map_url is None else map_url
    temp_deg_c = "" if temp_deg_c is None else temp_deg_c
    
    # Determine the new Sequence ID by adding 1 to the last existing Sequence ID
    last_sequence_id = df['Sequence ID'].max() if not df.empty else 100000
    new_sequence_id = int(last_sequence_id + 1)

    # Creating a list for the sequence column
    sequence = [new_sequence_id] * images_count
    sequence_numbers = list(range(1, images_count + 1))
    
    # Creating the 'File ID' column
    file_ids = [f"{new_sequence_id}_{num}" for num in sequence_numbers]

    # Creating a DataFrame with repeated values for each row
    new_data = pd.DataFrame({
        'File ID': file_ids,
        'Sequence ID': sequence,
        'Sequence': sequence_numbers,
        'Date': [img_date] * images_count,
        'Time': [img_time] * images_count,
        'Camera ID': [camera_id] * images_count,
        'Camera Make': [camera_make] * images_count,
        'Location': [location] * images_count,
        'GPS': [gps] * images_count,
        'Temperature': [temp_deg_c] * images_count,
        'Battery': [battery] * images_count,
        'SD Memory': [sd_memory] * images_count,
        'Map URL': [map_url] * images_count
    })
    
    # Converting 'Sequence' column to integer
    new_data['Sequence'] = new_data['Sequence'].astype(int)

    # Printing the summary information
    print(f"{current_time()} | Images: {images_count}, Camera ID: {camera_id}, Camera Make: {camera_make}"
        f"\n{current_time()} | Date: {img_date}, Time: {img_time}, Temperature: {temp_deg_c}℃"
        f"\n{current_time()} | Battery: {battery}%, SD Memory: {sd_memory}%"
        f"\n{current_time()} | Location: {location}, GPS: {gps}, Map URL: {map_url}"
    )

    # Append the new data to the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)
    
    return df




##############################################################################
############### ALERT CAPTION, ANNOTATE PICTURES, SEND ALERT #################
##############################################################################


from collections import Counter
import pandas as pd

def generate_alert_caption_en(df, human_warning, num_images, priority_detections, CLASSIFICATION_THRESHOLD):
    """
    Generate an alert caption based on the detection data in the dataframe.

    Parameters:
        - df (dataframe): The dataframe containing detection data.
        - human_warning (bool): Flag indicating if humans or vehicles were detected.
        - num_images (int): The number of images in the sequence.
        - PRIORITY_SPECIES (list): List of species that trigger priority alerts.
        - ALERT_LANGUAGE (str): The language for the alert ('en' or 'ro').
        - CLASSIFIER_CLASSES (list): List of classifier classes.
        - ROMANIAN_CLASSES (list): List of Romanian translations for classifier classes.

    Returns:
        - df (dataframe): The updated dataframe with the alert message.
        - alert_caption (str): The generated alert caption.
    """

    # Summary variables
    master_row = df.iloc[len(df) - num_images] # First row in the sequence
    sequence_id = master_row['Sequence ID']
    img_date = master_row['Date']
    img_time = master_row['Time']
    camera_id = master_row['Camera ID']
    camera_make = master_row['Camera Make']
    location = master_row['Location']
    gps = master_row['GPS']
    map_url = master_row['Map URL']
    temperature = master_row['Temperature']
    battery = master_row['Battery']
    sd_memory = master_row['SD Memory']

    # Image specific variables, stored as lists
    human_count = df['Human Count'].iloc[-num_images:].tolist()
    vehicle_count = df['Vehicle Count'].iloc[-num_images:].tolist()
    species_classes = df['Species Classes'].iloc[-num_images:].tolist()
    species_confidences = df['Species Confidences'].iloc[-num_images:].tolist() 

    alert_caption = ""

    if priority_detections:
        alert_caption += f"🚨🐻<b> Priority Species Detected </b>🐺🚨\n"

    if human_warning:
        alert_caption += f"🚶‍➡️🚜<b> Person Detected </b>🚜🚶\n"

    if not priority_detections and not human_warning:
        alert_caption += f"🦊🦌<b> Animal Detected </b>🦡🦉\n"

    alert_caption += f"\n"

    # Add human and vehicle counts to the alert caption with proper pluralization
    max_human_count = max(human_count)  # Get the max number of humans detected in any one image
    max_vehicle_count = max(vehicle_count)  # Get the max number of vehicles detected in any one image

    if max_human_count > 0:  # If any humans detected
        human_label = "Person" if max_human_count == 1 else "People"
        alert_caption += f"🔸 {max_human_count:.0f} {human_label}\n"
    
    if max_vehicle_count > 0:  # If any vehicles detected
        vehicle_label = "Vehicle" if max_vehicle_count == 1 else "Vehicles"
        alert_caption += f"🔸 {max_vehicle_count:.0f} {vehicle_label}\n"


    # Initialize variables
    species_count = Counter()  # Counter to hold species count
    species_confidences_dict = {}  # Dictionary to hold species and their confidence ranges

    # Iterate through each image and process the species and confidence data
    for i in range(num_images):
        image_species = species_classes[i]  # Species detected in the image
        image_confidences = species_confidences[i]  # Confidence for the species detected

        # Count the occurrences of each species
        for species, confidence in zip(image_species, image_confidences):
            species_count[species] = max(species_count.get(species, 0), image_species.count(species))

            # Store the min and max confidence for each species
            if species not in species_confidences_dict:
                species_confidences_dict[species] = [confidence, confidence]
            else:
                species_confidences_dict[species][0] = min(species_confidences_dict[species][0], confidence)  # Min confidence
                species_confidences_dict[species][1] = max(species_confidences_dict[species][1], confidence)  # Max confidence

    # Add the animals detected to the the alert caption.

    for species, count in species_count.items():
        min_conf, max_conf = species_confidences_dict[species]

        # Skip species if the max confidence is below the threshold
        if max_conf < CLASSIFICATION_THRESHOLD:
            continue  # Skip adding this species to the alert if the max confidence is below threshold

        # Handle plural species correctly
        species_label = species if count == 1 else f"{species}s"  # Add "s" for plural species

        # Format the alert line for the species
        if round(min_conf, 2) == round(max_conf, 2):
            # If min and max confidence are the same, just show the single confidence value
            alert_caption += f"🔹 {count} {species_label} ({min_conf*100:.0f}% confidence)\n"
        else:
            # If min and max confidence are different, show the confidence range
            alert_caption += f"🔹 {count} {species_label} ({min_conf*100:.0f}-{max_conf*100:.0f}% confidence)\n"

    alert_caption += f"\n"

    if human_warning:
        alert_caption += f"❗️ Do not store or distribute photos of people without authorisation.\n\n"

    # Date/Time
    if img_date and img_time:
        alert_caption += f"🕔 Date/Time: {img_date} | {img_time}\n"

    # Location
    if location:
        alert_caption += f"🌍 Location: {location}\n"

    # Map Link
    if map_url:
        alert_caption += f"📍 <a href='{map_url}'>Map Link</a>\n"

    alert_caption += f"\n⚙️ "

    # Camera ID, Brand, Battery, Storage, etc.
    if camera_id:
        alert_caption += f"Camera ID: {camera_id} | "
    if camera_make:
        alert_caption += f"Brand: {camera_make} | "
    if battery:
        alert_caption += f"Battery: {battery:.0f}% | "
    if sd_memory:
        alert_caption += f"Storage: {sd_memory:.0f}% | "
    if temperature:
        alert_caption += f"Temperature: {temperature}℃ | "
    if gps:
        alert_caption += f"{gps} | "
    if sequence_id:
        alert_caption += f"File ID: {int(sequence_id)}"

    # Store the alert caption in the dataframe
    df.iloc[-num_images:, df.columns.get_loc('Alert Message')] = alert_caption.replace('\n', '. ')


    return df, alert_caption


def generate_alert_caption_ro(df, human_warning, num_images, priority_detections, CLASSIFICATION_THRESHOLD, CLASSIFIER_CLASSES, ROMANIAN_CLASSES):
    """
    Generate an alert caption based on the detection data in the dataframe.

    Parameters:
        - df (dataframe): The dataframe containing detection data.
        - human_warning (bool): Flag indicating if humans or vehicles were detected.
        - num_images (int): The number of images in the sequence.
        - PRIORITY_SPECIES (list): List of species that trigger priority alerts.
        - ALERT_LANGUAGE (str): The language for the alert ('en' or 'ro').
        - CLASSIFIER_CLASSES (list): List of classifier classes.
        - ROMANIAN_CLASSES (list): List of Romanian translations for classifier classes.

    Returns:
        - df (dataframe): The updated dataframe with the alert message.
        - alert_caption (str): The generated alert caption.
    """

    # Summary variables
    master_row = df.iloc[len(df) - num_images] # First row in the sequence
    sequence_id = master_row['Sequence ID']
    img_date = master_row['Date']
    img_time = master_row['Time']
    camera_id = master_row['Camera ID']
    camera_make = master_row['Camera Make']
    location = master_row['Location']
    gps = master_row['GPS']
    map_url = master_row['Map URL']
    temperature = master_row['Temperature']
    battery = master_row['Battery']
    sd_memory = master_row['SD Memory']

    # Image specific variables, stored as lists
    human_count = df['Human Count'].iloc[-num_images:].tolist()
    vehicle_count = df['Vehicle Count'].iloc[-num_images:].tolist()
    species_classes = df['Species Classes'].iloc[-num_images:].tolist()
    species_confidences = df['Species Confidences'].iloc[-num_images:].tolist() 

    alert_caption = ""

    if priority_detections:
        alert_caption += f"🚨🐻<b> Specie Prioritară Detectată </b>🐺🚨\n"

    if human_warning:
        alert_caption += f"🚶‍➡️🚜<b> Persoană Detectată </b>🚜🚶\n"

    if not priority_detections and not human_warning:
        alert_caption += f"🦊🦌<b> Animal Detectat </b>🦡🦉\n"

    alert_caption += f"\n"

    # Add human and vehicle counts to the alert caption with proper pluralization
    max_human_count = max(human_count)  # Get the max number of humans detected in any one image
    max_vehicle_count = max(vehicle_count)  # Get the max number of vehicles detected in any one image

    if max_human_count > 0:  # If any humans detected
        human_label = "Persoană" if max_human_count == 1 else "Persoane"
        alert_caption += f"🔸 {max_human_count:.0f} {human_label}\n"
    
    if max_vehicle_count > 0:  # If any vehicles detected
        vehicle_label = "Vehicul" if max_vehicle_count == 1 else "Vehicule"
        alert_caption += f"🔸 {max_vehicle_count:.0f} {vehicle_label}\n"


    # Initialize variables
    species_count = Counter()  # Counter to hold species count
    species_confidences_dict = {}  # Dictionary to hold species and their confidence ranges

    # Iterate through each image and process the species and confidence data
    for i in range(num_images):
        image_species = species_classes[i]  # Species detected in the image
        image_confidences = species_confidences[i]  # Confidence for the species detected

        # Count the occurrences of each species
        for species, confidence in zip(image_species, image_confidences):
            species_count[species] = max(species_count.get(species, 0), image_species.count(species))

            # Store the min and max confidence for each species
            if species not in species_confidences_dict:
                species_confidences_dict[species] = [confidence, confidence]
            else:
                species_confidences_dict[species][0] = min(species_confidences_dict[species][0], confidence)  # Min confidence
                species_confidences_dict[species][1] = max(species_confidences_dict[species][1], confidence)  # Max confidence

    # Helper function to pluralise Romanian animal names
    def pluralize_romanian(word, count):
        if count > 1:
            if word == "Cal":
                return "Cai"
            elif word.endswith("ă"):
                return word[:-1] + "e"
            elif word.endswith('e'):
                return word[:-1] + "i"
            else:
                return word + "i"
        return word

    # Add the animals detected to the the alert caption.

    for species, count in species_count.items():
        min_conf, max_conf = species_confidences_dict[species]

        # Skip species if the max confidence is below the threshold
        if max_conf < CLASSIFICATION_THRESHOLD:
            continue  # Skip adding this species to the alert if the max confidence is below threshold

        index = CLASSIFIER_CLASSES.index(species)
        species_romanian = ROMANIAN_CLASSES[index]
        
        # Handle plural species correctly in Romanian
        species_label = pluralize_romanian(species_romanian, count)

        # Format the alert line for the species
        if round(min_conf, 2) == round(max_conf, 2):
            # If min and max confidence are the same, just show the single confidence value
            alert_caption += f"🔹 {count} {species_label} ({min_conf*100:.0f}% precizie)\n"
        else:
            # If min and max confidence are different, show the confidence range
            alert_caption += f"🔹 {count} {species_label} ({min_conf*100:.0f}-{max_conf*100:.0f}% precizie)\n"

    alert_caption += f"\n"

    if human_warning:
        alert_caption += f"❗️ Pentru a proteja intimitatea, nu distribuiți poze cu oameni în afara organizației.\n\n"

    # Date/Time
    if img_date and img_time:
        alert_caption += f"🕔 Dată/Ora: {img_date} | {img_time}\n"

    # Location
    if location:
        alert_caption += f"🌍 Toponim: {location}\n"

    # Map Link
    if map_url:
        alert_caption += f"📍 <a href='{map_url}'>Arată pe hartă</a>\n"

    alert_caption += f"\n⚙️ "

    # Camera ID, Brand, Battery, Storage, etc.
    if camera_id:
        alert_caption += f"Cameră: {camera_id} | "
    if camera_make:
        alert_caption += f"Marcă: {camera_make} | "
    if battery:
        alert_caption += f"Baterie: {battery:.0f}% | "
    if sd_memory:
        alert_caption += f"Spațiu pe card: {sd_memory:.0f}% | "
    if temperature:
        alert_caption += f"Temperatură: {temperature}℃ | "
    if gps:
        alert_caption += f"{gps} | "
    if sequence_id:
        alert_caption += f"ID: {int(sequence_id)}"

    # Store the alert caption in the dataframe
    df.iloc[-num_images:, df.columns.get_loc('Alert Message')] = alert_caption.replace('\n', '. ')


    return df, alert_caption

from PIL import Image, ImageDraw, ImageFont
import io
from IPython.display import display

def annotate_images_en(df, images, CLASSIFICATION_THRESHOLD):
    """
    Annotates images with bounding boxes and labels based on detection data. This function ensures that only animals with 
    a confidence above a given threshold are annotated, but it will annotate animals even if their confidence is below 
    the threshold in the current image if they are above the threshold in any other image within the sequence. 
    The function handles animal, human, and vehicle detections and allows for dynamic font size adjustment for labels.

    Parameters:
        - df (dataframe): The dataframe containing detection data for each image in the sequence. It should include 
                           the following columns:
                           - 'Detection Boxes' (list of bounding box coordinates for each detection),
                           - 'Detection Classes' (list indicating the type of detection: animal, human, vehicle),
                           - 'Detection Confidences' (list of confidence values for each detection),
                           - 'Species Classes' (list of species names for animal detections),
                           - 'Species Confidences' (list of confidence values for each species detection).
        
        - images (list): A list of images (PIL Image objects) to be annotated. The list should be in the same order 
                         as the rows in `df`.
        
        - CLASSIFICATION_THRESHOLD (float): The confidence threshold (between 0 and 1) for annotating animals. Detections 
                                              with confidence below this threshold will be skipped, unless they meet the 
                                              threshold in any other image in the sequence.

    Returns:
        - image_list (list): A list of annotated images (PIL Image objects) with bounding boxes and labels drawn 
                             for detections that meet the threshold criteria.

    Function Details:
        - The function first processes all images in the sequence to track species confidence values across all images 
          in the sequence.
        
        - It then iterates through each image, drawing bounding boxes around detections and labeling them based on their 
          class (animal, human, or vehicle) and confidence.
        
        - For animals, if the species confidence in the current image is below the threshold, it will only be annotated 
          if its confidence in any other image in the sequence is above the threshold.
        
        - The bounding boxes are drawn in different colors based on the detection class:
            - Red for animals
            - Green for humans
            - Blue for vehicles
            - Yellow for unknown detections
        
        - The labels are dynamically sized based on the image height to ensure visibility, and the text is placed above 
          the bounding boxes with a contrasting background to ensure readability.
        
        - The function returns a list of annotated images, or `None` if there are no valid detections that meet the 
          specified threshold.

    Example:
        If the `CLASSIFICATION_THRESHOLD` is set to 0.8, animals with a confidence score greater than or equal to 80% 
        will be annotated. If an animal is detected below this threshold in one image but above it in another, the animal 
        will still be annotated.
    """

    image_list = []
    
    # Extract the last len(images) rows
    last_rows = df.iloc[-len(images):]

    # Track the species confidences across the entire image sequence
    all_species_confidences = {}

    # Store species confidences across the images in the sequence
    for _, row_data in last_rows.iterrows():
        species_classes = row_data['Species Classes']
        species_confidences = row_data['Species Confidences']
        
        for species, species_conf in zip(species_classes, species_confidences):
            if species not in all_species_confidences:
                all_species_confidences[species] = []
            all_species_confidences[species].append(species_conf)


    # Now annotate the images
    for image, row in zip(images, last_rows.iterrows()):
        index, row_data = row
        draw = ImageDraw.Draw(image)

        detection_boxes = row_data['Detection Boxes']
        detection_classes = row_data['Detection Classes']
        detection_confidences = row_data['Detection Confidences']
        species_classes = row_data['Species Classes']
        species_confidences = row_data['Species Confidences']
        
        species_idx = 0  # to keep track of the current species index
        width, height = image.size
        font_size = max(10, int(height * 0.02))  # font size is 2% of the image height, with a minimum of 10
        font = ImageFont.truetype("../Ubuntu-B.ttf", font_size)

        for box, d_class, d_conf in zip(detection_boxes, detection_classes, detection_confidences):

            x1, y1, x2, y2 = map(int, box)  # Convert directly to integers

            # Set color and label based on detection class
            if d_class == 0:  # Animal

                species_label = species_classes[species_idx]
                species_confidence = species_confidences[species_idx]

                # If the species confidence is below the threshold in the current image, skip it unless it is above the threshold in other images
                if species_confidence < CLASSIFICATION_THRESHOLD and all(species_conf < CLASSIFICATION_THRESHOLD for species_conf in all_species_confidences[species_label]):
                    species_idx += 1
                    continue

                color = 'red'
                label = f"Animal {d_conf * 100:.0f}% | {species_label} {species_confidences[species_idx] * 100:.0f}%"
                species_idx += 1

            elif d_class == 1:  # Human
                color = 'green'
                label = f"Human {d_conf * 100:.0f}%"

            elif d_class == 2:  # Vehicle
                color = 'blue'
                label = f"Vehicle {d_conf * 100:.0f}%"
                    
            else:
                color = 'yellow'
                label = f"Unknown {d_conf * 100:.0f}%"
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Get text size using textbbox
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_background = [x1, y1 - text_height, x1 + text_width, y1]

            # Draw text background
            draw.rectangle(text_background, fill=color)
            
            # Draw text with a dynamic offset for alignment
            offset = int(font_size * 0.2)  # Dynamic offset based on the font size
            draw.text((x1, y1 - text_height - offset), label, fill="white", font=font)
        
        image_list.append(image)
    
    return image_list


def annotate_images_ro(df, images, CLASSIFICATION_THRESHOLD, CLASSIFIER_CLASSES, ROMANIAN_CLASSES):
    """
    Annotates images with bounding boxes and labels based on detection data. This function ensures that only animals with 
    a confidence above a given threshold are annotated, but it will annotate animals even if their confidence is below 
    the threshold in the current image if they are above the threshold in any other image within the sequence. 
    The function handles animal, human, and vehicle detections and allows for dynamic font size adjustment for labels.

    Parameters:
        - df (dataframe): The dataframe containing detection data for each image in the sequence. It should include 
                           the following columns:
                           - 'Detection Boxes' (list of bounding box coordinates for each detection),
                           - 'Detection Classes' (list indicating the type of detection: animal, human, vehicle),
                           - 'Detection Confidences' (list of confidence values for each detection),
                           - 'Species Classes' (list of species names for animal detections),
                           - 'Species Confidences' (list of confidence values for each species detection).
        
        - images (list): A list of images (PIL Image objects) to be annotated. The list should be in the same order 
                         as the rows in `df`.
        
        - CLASSIFICATION_THRESHOLD (float): The confidence threshold (between 0 and 1) for annotating animals. Detections 
                                              with confidence below this threshold will be skipped, unless they meet the 
                                              threshold in any other image in the sequence.

    Returns:
        - image_list (list): A list of annotated images (PIL Image objects) with bounding boxes and labels drawn 
                             for detections that meet the threshold criteria.

    Function Details:
        - The function first processes all images in the sequence to track species confidence values across all images 
          in the sequence.
        
        - It then iterates through each image, drawing bounding boxes around detections and labeling them based on their 
          class (animal, human, or vehicle) and confidence.
        
        - For animals, if the species confidence in the current image is below the threshold, it will only be annotated 
          if its confidence in any other image in the sequence is above the threshold.
        
        - The bounding boxes are drawn in different colors based on the detection class:
            - Red for animals
            - Green for humans
            - Blue for vehicles
            - Yellow for unknown detections
        
        - The labels are dynamically sized based on the image height to ensure visibility, and the text is placed above 
          the bounding boxes with a contrasting background to ensure readability.
        
        - The function returns a list of annotated images, or `None` if there are no valid detections that meet the 
          specified threshold.

    Example:
        If the `CLASSIFICATION_THRESHOLD` is set to 0.8, animals with a confidence score greater than or equal to 80% 
        will be annotated. If an animal is detected below this threshold in one image but above it in another, the animal 
        will still be annotated.
    """

    image_list = []
    
    # Extract the last len(images) rows
    last_rows = df.iloc[-len(images):]

    # Track the species confidences across the entire image sequence
    all_species_confidences = {}

    # Store species confidences across the images in the sequence
    for _, row_data in last_rows.iterrows():
        species_classes = row_data['Species Classes']
        species_confidences = row_data['Species Confidences']
        
        for species, species_conf in zip(species_classes, species_confidences):
            if species not in all_species_confidences:
                all_species_confidences[species] = []
            all_species_confidences[species].append(species_conf)


    # Now annotate the images
    for image, row in zip(images, last_rows.iterrows()):
        index, row_data = row
        draw = ImageDraw.Draw(image)

        detection_boxes = row_data['Detection Boxes']
        detection_classes = row_data['Detection Classes']
        detection_confidences = row_data['Detection Confidences']
        species_classes = row_data['Species Classes']
        species_confidences = row_data['Species Confidences']
        
        species_idx = 0  # to keep track of the current species index
        width, height = image.size
        font_size = max(10, int(height * 0.02))  # font size is 2% of the image height, with a minimum of 10
        font = ImageFont.truetype("../Ubuntu-B.ttf", font_size)

        for box, d_class, d_conf in zip(detection_boxes, detection_classes, detection_confidences):

            x1, y1, x2, y2 = map(int, box)  # Convert directly to integers
            
            # Set color and label based on detection class
            if d_class == 0:  # Animal

                species_label = species_classes[species_idx]
                species_confidence = species_confidences[species_idx]

                # Translate the species label to Romanian
                index = CLASSIFIER_CLASSES.index(species_label)
                species_label_romanian = ROMANIAN_CLASSES[index]

                # If the species confidence is below the threshold in the current image, skip it unless it is above the threshold in other images
                if species_confidence < CLASSIFICATION_THRESHOLD and all(species_conf < CLASSIFICATION_THRESHOLD for species_conf in all_species_confidences[species_label]):
                    species_idx += 1
                    continue

                color = 'red'
                label = f"Animal {d_conf * 100:.0f}% | {species_label_romanian} {species_confidences[species_idx] * 100:.0f}%"
                species_idx += 1

            elif d_class == 1:  # Human
                color = 'green'
                label = f"Persoană {d_conf * 100:.0f}%"

            elif d_class == 2:  # Vehicle
                color = 'blue'
                label = f"Vehicul {d_conf * 100:.0f}%"
                    
            else:
                color = 'yellow'
                label = f"Necunoscut {d_conf * 100:.0f}%"
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Get text size using textbbox
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_background = [x1, y1 - text_height, x1 + text_width, y1]

            # Draw text background
            draw.rectangle(text_background, fill=color)
            
            # Draw text with a dynamic offset for alignment
            offset = int(font_size * 0.2)  # Dynamic offset based on the font size
            draw.text((x1, y1 - text_height - offset), label, fill="white", font=font)
        
        image_list.append(image)
    
    return image_list


import requests
import io
import json
import time
from requests.exceptions import ConnectionError, HTTPError

def send_alert_to_telegram(bot_token, chat_id, photos, caption):
    """
    Send an alert with photos and a caption to the Telegram group.

    Parameters:
        - bot_token (str): The bot token for authenticating with the Telegram API.
        - chat_id (str): The ID of the chat to send the alert to.
        - photos (list): A list of photos to be sent. If None or empty, only the caption will be sent.
        - caption (str): The caption to be included with the photos.

    Returns:
        - None

    Raises:
        - Exception: If the maximum number of retries is reached when sending the request.
        - ValueError: If any file to be sent is empty.
    """
    def prepare_files(photos):
        media = []
        files = {}
        for idx, photo in enumerate(photos):
            buf = io.BytesIO()
            photo.save(buf, format='JPEG')
            buf.seek(0)
            file_name = f'photo{idx}.jpg'
            files[file_name] = buf.getvalue()  # Store the bytes data
            media.append({
                'type': 'photo',
                'media': f'attach://{file_name}',
                'caption': caption if idx == 0 else "",
                'parse_mode': 'HTML'  # Enable HTML parsing for each media item
            })
        return media, files

    def send_request(url, data, files, retries=0, MAX_RETRIES=5):
        if retries >= MAX_RETRIES:
            raise Exception("Max retries reached")

        try:
            # Convert files dictionary to the required format for requests
            files_prepared = {name: (name, io.BytesIO(buf), 'image/jpeg') for name, buf in files.items()}
            response = requests.post(url, data=data, files=files_prepared)
            if response.status_code == 429:
                # Handle rate limit
                retry_after = int(response.headers.get("Retry-After", 1))
                print(f"ATTEMPT {retries + 1}. Error response:", response.json())  # Print the error response for debugging
                retry_after = retry_after + 10
                print(f"Rate limited. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                return send_request(url, data, files, retries + 1)  # Retry after delay
            if response.status_code != 200:
                print(f"ATTEMPT {retries + 1}. Error response:", response.json())  # Print the error response for debugging
                print("Request data:", data)  # Print request data for debugging
                if files:
                    print("Request files:", files)  # Print request files for debugging
            response.raise_for_status()
            return response
        except (ConnectionError, HTTPError) as e:
            print(f"ATTEMPT {retries + 1}. Connection error: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # Wait for 10 seconds before retrying
            return send_request(url, data, files, retries + 1)

    if photos is None or len(photos) == 0:
        # URL for sending text message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': caption,
            'parse_mode': 'HTML'  # Enable HTML parsing
        }
        files = {}
        response = send_request(url, data, files)
    else:
        # URL for sending media group
        url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"
        media, files = prepare_files(photos)

        # Prepare the data for the request
        data = {
            'chat_id': chat_id,
            'media': json.dumps(media),
            'parse_mode': 'HTML'  # Ensure HTML parsing is enabled globally
        }

        # Ensure files are not empty
        for file_name, file_content in files.items():
            if len(file_content) == 0:
                raise ValueError(f"The file {file_name} is empty.")

        response = send_request(url, data, files)




##################################################
############### HELPER FUNCTIONS #################
##################################################

def current_time():
    """
    Get the current time formatted as a string.

    Returns:
        - now (str): The current time formatted as "YYYY-MM-DD HH:MM:SS".
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return now


from datetime import datetime
import pytz

def get_current_time_in_romania():
    """
    Get the current time in Romania.

    Returns:
        - (str): The current time in Romania formatted as "HH:MM".
    """
    romania_tz = pytz.timezone('Europe/Bucharest')
    return datetime.now(romania_tz).strftime("%H:%M")

import os
from PIL import Image


def save_images(df, images, human_warning, PHOTOS_PATH, CLASSIFICATION_THRESHOLD):
    """
    Save images to a specified directory and update the dataframe with the file paths.

    Parameters:
        - df (dataframe): The dataframe containing detection data.
        - images (list): A list of images to be saved.
        - human_warning (bool): Flag indicating if humans or vehicles were detected.
        - PHOTOS_PATH (str): The path to the directory where images will be saved.

    Returns:
        - df (dataframe): The updated dataframe with file paths for the saved images.
    """
    # Ensure the PHOTOS_PATH directory exists
    os.makedirs(PHOTOS_PATH, exist_ok=True)

    # Get the last len(images) rows of df
    last_rows = df.tail(len(images))
    
    # Iterate through the images and corresponding rows in the DataFrame
    for index, (image, (_, row)) in enumerate(zip(images, last_rows.iterrows())):
        primary_species = row['Primary Species']
        file_id = row['File ID']

        # Check if any detection's confidence is above the threshold
        if any(species_conf >= CLASSIFICATION_THRESHOLD for species_conf in row['Species Confidences']):

            # Create directory for the primary species if it doesn't exist
            if human_warning:
                species_dir = os.path.join(PHOTOS_PATH, "Human")
            else:
                species_dir = os.path.join(PHOTOS_PATH, primary_species)
            
            os.makedirs(species_dir, exist_ok=True)

            # Define the file path for the image
            file_path = os.path.join(species_dir, f"{file_id}.jpg")

            # Save the image
            image.save(file_path, format='JPEG')

            print(f"{current_time()} | Image {index + 1} saved at: {file_path}")

            # Update the File Path column in the DataFrame
            df.at[row.name, 'File Path'] = file_path

    return df


import yaml # Handles YAML (config) files


# Function to load the configuration from the file and return values
def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    
    try:

        # Extract system settings from the config file
        detection_threshold = float(config['system_config']['DETECTION_THRESHOLD'])
        detector_model_v6 = bool(config['system_config']['DETECTOR_MODEL_V6'])
        classification_threshold = float(config['system_config']['CLASSIFICATION_THRESHOLD'])
        alert_language = str(config['system_config']['LANGUAGE'].lower())  # Convert to lowercase
        priority_species = config['system_config']['PRIORITY_SPECIES']
        check_email_frequency = int(config['system_config']['CHECK_EMAIL_FREQUENCY'])

        # Extract email settings (IMAP)
        imap_host = config['imap_config']['host']
        email_user = config['imap_config']['user']
        email_pass = config['imap_config']['password']

        # Extract Telegram settings
        telegram_bot_token = config['telegram_config']['bot_token']
        telegram_chat_id_all = config['telegram_config']['chat_id_all']
        telegram_chat_id_priority = config['telegram_config']['chat_id_priority']
        telegram_chat_id_human = config['telegram_config']['chat_id_human']

        # Extract SMTP settings
        smtp_server = config['smtp_config']['host']
        smtp_port = int(config['smtp_config']['port'])
        email_sender = config['smtp_config']['user']
        email_password = config['smtp_config']['password']
        recipients = config['smtp_config']['recipients']

        last_config_mod_time = os.path.getmtime(CONFIG_PATH)

        # Print a message showing the settings have been loaded
        print(f"{current_time()} | Current Settings:  "
            f"DETECTION_THRESHOLD={detection_threshold}, CLASSIFICATION_THRESHOLD={classification_threshold}, "
            f"ALERT_LANGUAGE={alert_language}, CHECK_EMAIL_FREQUENCY={check_email_frequency} seconds, "
            f"PRIORITY_SPECIES={priority_species}")

        # Return all settings as a dictionary or tuple
        return {
            'DETECTION_THRESHOLD': detection_threshold,
            'DETECTOR_MODEL_V6': detector_model_v6,
            'CLASSIFICATION_THRESHOLD': classification_threshold,
            'ALERT_LANGUAGE': alert_language,
            'PRIORITY_SPECIES': priority_species,
            'CHECK_EMAIL_FREQUENCY': check_email_frequency,
            'IMAP_HOST': imap_host,
            'EMAIL_USER': email_user,
            'EMAIL_PASS': email_pass,
            'TELEGRAM_BOT_TOKEN': telegram_bot_token,
            'TELEGRAM_CHAT_ID_ALL': telegram_chat_id_all,
            'TELEGRAM_CHAT_ID_PRIORITY': telegram_chat_id_priority,
            'TELEGRAM_CHAT_ID_HUMAN': telegram_chat_id_human,
            'SMTP_SERVER': smtp_server,
            'SMTP_PORT': smtp_port,
            'EMAIL_SENDER': email_sender,
            'EMAIL_PASSWORD': email_password,
            'RECIPIENTS': recipients,
            'last_config_mod_time': last_config_mod_time
        }

    except Exception as e:
        # Handle errors gracefully
        print(f"{current_time()} | SETTINGS NOT UPDATED. Error loading config file: {e}")
