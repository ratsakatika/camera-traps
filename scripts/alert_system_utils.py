#################################################
############ IMAGE PROCESSING TOOLS #############
#################################################

import torch
import timm
from datetime import datetime
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from megadetector.visualization import visualization_utils as vis_utils
import io
from PIL import Image
import ast
from IPython.display import display
from megadetector.visualization import visualization_utils as vis_utils

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def image_to_bytes(image):
    """Convert an image to bytes."""
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr

def detector(df, model, images, DETECTION_THRESHOLD):
    """Run the MegaDetector on images and return detections above the threshold."""
    
    human_warning = False
    num_images = len(images)
    
    # Ensure there are enough rows in the DataFrame to update
    if len(df) < num_images:
        raise ValueError("The DataFrame does not have enough rows to update.")

    for i, image in enumerate(images):
        processed_image = vis_utils.load_image(image_to_bytes(image))
        result = model.generate_detections_one_image(processed_image)
        detections_above_threshold = [d for d in result['detections'] if d['conf'] > DETECTION_THRESHOLD]

        detection_boxes = [d['bbox'] for d in detections_above_threshold]
        detection_classes = [d['category'] for d in detections_above_threshold]
        detection_confidences = [d['conf'] for d in detections_above_threshold]

        animal_count = sum(1 for d in detections_above_threshold if d['category'] == '1')
        human_count = sum(1 for d in detections_above_threshold if d['category'] == '2')
        vehicle_count = sum(1 for d in detections_above_threshold if d['category'] == '3')

        print(f"{current_time()} | Image {i+1}: Animal Count = {animal_count}, Human Count = {human_count}, Vehicle Count = {vehicle_count}")

        # Update the respective row in the DataFrame
        df.at[df.index[-num_images + i], 'Detection Boxes'] = detection_boxes
        df.at[df.index[-num_images + i], 'Detection Classes'] = detection_classes
        df.at[df.index[-num_images + i], 'Detection Confidences'] = detection_confidences
        df.at[df.index[-num_images + i], 'Animal Count'] = animal_count
        df.at[df.index[-num_images + i], 'Human Count'] = human_count
        df.at[df.index[-num_images + i], 'Vehicle Count'] = vehicle_count

        if human_count > 0 or vehicle_count > 0:
            human_warning = True
            

    return df, human_warning


class classifier:
    """Image classifier for animal species."""
    def __init__(self, model_path_classifier, backbone, animal_classes, device='cpu'):
        self.model = timm.create_model(backbone, pretrained=False, num_classes=len(animal_classes), dynamic_img_size=True)
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

from collections import Counter

def batch_classification(df, classifier_model, images, CLASSIFICATION_THRESHOLD):
    num_images = len(images)
    df_length = len(df)

    for i, image in enumerate(images):

        species_list = []
        species_confidence_list = []
        detection_boxes = df['Detection Boxes'][df_length-num_images + i]

        if detection_boxes == []:

            primary_species = "Empty"
            wild_boar_count = 0
            bear_count = 0
            print(f"{current_time()} | Image {i+1}, No Detections")

        else:

            detection_classes = df['Detection Classes'][df_length-num_images + i]
            detection_conf = df['Detection Confidences'][df_length-num_images + i]

            for j, bbox in enumerate(detection_boxes):
                
                if detection_classes[j] == '1':  # Only classify if an animal

                    left, top, width, height = bbox  # Unpack the bounding box

                    # Calculate the cropping coordinates
                    left_resized = int(left * image.width)
                    top_resized = int(top * image.height)
                    right_resized = int((left + width) * image.width)
                    bottom_resized = int((top + height) * image.height)

                    # Ensure the coordinates are within the image boundaries
                    left_resized = max(0, min(left_resized, image.width))
                    top_resized = max(0, min(top_resized, image.height))
                    right_resized = max(0, min(right_resized, image.width))
                    bottom_resized = max(0, min(bottom_resized, image.height))

                    cropped_image = image.crop((left_resized, top_resized, right_resized, bottom_resized))

                    species, species_conf = classifier_model.predict(cropped_image)

                    if species_conf >= CLASSIFICATION_THRESHOLD:
                        
                        species_list.append(species)
                        species_confidence_list.append(species_conf)

                        print(f"{current_time()} | Image {i+1}, Detection {j+1} ({detection_conf[j] * 100:.2f}% confidence), Species: {species} ({species_conf * 100:.2f}% confidence)")
                    
                    else:

                        species_list.append("Unknown")
                        species_confidence_list.append(species_conf)
                        print(f"{current_time()} | Image {i+1}, Detection {j+1} ({detection_conf[j] * 100:.2f}% confidence), Species: {species} ({species_conf * 100:.2f}% confidence). CLASSED AS 'UNKNOWN': Below Species Confidence Threshold ({CLASSIFICATION_THRESHOLD * 100:.2f}%)")

                    
                    # display(cropped_image)

            if species_list:

                # Count the occurrences of each species
                species_counter = Counter(species_list)
                most_common_count = max(species_counter.values())

                # Identify species with the highest count
                most_common_species = [species for species, count in species_counter.items() if count == most_common_count]

                #If there's a tie, select the species with the highest confidence
                if len(most_common_species) > 1:
                    max_confidence = -1
                    primary_species = None
                    for species in most_common_species:
                        indices = [i for i, x in enumerate(species_list) if x == species]
                        max_species_confidence = max(species_confidence_list[i] for i in indices)
                        if max_species_confidence > max_confidence:
                            max_confidence = max_species_confidence
                            primary_species = species
                else:
                    primary_species = most_common_species[0]

                wild_boar_count = species_list.count('Wild Boar')
                bear_count = species_list.count('Bear')

            elif df['Human Count'][df_length-num_images + i] > 0:
                print(f"{current_time()} | Image {i+1}, Human/Vehicle Only")
                primary_species = "Human"
                wild_boar_count = 0
                bear_count = 0

            elif df['Vehicle Count'][df_length-num_images + i] > 0:
                print(f"{current_time()} | Image {i+1}, Human/Vehicle Only")
                primary_species = "Vehicle"
                wild_boar_count = 0
                bear_count = 0
            
            else:
                print(f"{current_time()} | Image {i+1}, Error")
                primary_species = "Error"
                wild_boar_count = 0
                bear_count = 0

        df.at[df.index[-num_images + i], 'Species Classes'] = species_list
        df.at[df.index[-num_images + i], 'Species Confidences'] = species_confidence_list
        df.at[df.index[-num_images + i], 'Primary Species'] = primary_species
        df.at[df.index[-num_images + i], 'Wild Boar Count'] = wild_boar_count
        df.at[df.index[-num_images + i], 'Bear Count'] = bear_count

    return df


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
    original_images = []
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
        original_images = [img.copy() for img in images]
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

    return images, original_images, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory


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
        print(f"{current_time()} | Error downloading image from {url}: {str(e)}")
        return None


def extract_camera_id(subject, body):
    # Extract camera ID from subject
    camera_id = "Unknown"

    if subject and '_' in subject:
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
        print(f"{current_time()} | Error extracting date and time from image: {e}")
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

def extract_and_update_camera_info(CAMERA_LOCATIONS_PATH, camera_id, battery=None, sd_memory=None):
    # Read the CSV file
    df = pd.read_csv(CAMERA_LOCATIONS_PATH)

    # Find the row with the matching Camera ID
    camera_row = df[df['Camera ID'] == camera_id]

    if camera_row.empty:
        print(f"{current_time()} | No camera found with ID: {camera_id}")
        return None, None, None, None, None, None

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
    
    if camera_make == None:
        camera_id = "Unknown"
        camera_make = "Unknown"
        location = "Unknown"
        gps = "Unknown"
        battery = "Unknown"
        sd_memory = "Unknown"
        map_url = "Unknown"
        temp_deg_c = "Unknown"
    
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
    print(
        f"{current_time()} | *** Email Received ***"
        f"\n{current_time()} | Images: {images_count}, Camera ID: {camera_id}, Camera Make: {camera_make}"
        f"\n{current_time()} | Date: {img_date}, Time: {img_time}, Temperature: {temp_deg_c}"
        f"\n{current_time()} | Battery: {battery}%, SD Memory: {sd_memory}%"
        f"\n{current_time()} | Location: {location}, GPS: {gps}, Map URL: {map_url}"
    )

    # Append the new data to the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)
    
    return df

def current_time():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return now


##############################################################################
############### ALERT CAPTION, ANNOTATE PICTURES, SEND ALERT #################
##############################################################################


from collections import Counter
import pandas as pd

def generate_alert_caption(df, human_warning, num_images, SPECIES_OF_INTEREST, EMAIL_USER, ALERT_LANGUAGE="en"):
    

    alert_caption = ""

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
    primary_species = df['Primary Species'].iloc[-num_images:].tolist()
    animal_count = df['Animal Count'].iloc[-num_images:].tolist()
    wild_boar_count = df['Wild Boar Count'].iloc[-num_images:].tolist()
    bear_count = df['Bear Count'].iloc[-num_images:].tolist()
    human_count = df['Human Count'].iloc[-num_images:].tolist()
    vehicle_count = df['Vehicle Count'].iloc[-num_images:].tolist()
    species_classes = df['Species Classes'].iloc[-num_images:].tolist()
    species_confidences = df['Species Confidences'].iloc[-num_images:].tolist() 

    # Determine if a priority alert should be sent (e.g. wild boar or bear detected)
    priority_alert = None

    for species_list in species_classes:
        for species in SPECIES_OF_INTEREST:
            if species in species_list:
                priority_alert = species
                break
        if priority_alert:
            break

    if priority_alert == "Wild Boar":
        priority_alert_count = max(wild_boar_count)
    elif priority_alert == "Bear":
        priority_alert_count = max(bear_count)



    most_common_species = ""
    # Remove "Empty", "Human", and "Vehicle"
    primary_species_modified = [species for species in primary_species if species not in {"Empty", "Human", "Vehicle"}]
    species_counts = Counter(primary_species_modified)

    if species_counts:

        max_count = max(species_counts.values())
        most_common_species = [species for species, count in species_counts.items() if count == max_count]

        if len(most_common_species) == 1:
            sequence_primary_species = (most_common_species[0])
        else:
            sequence_primary_species = ", ".join(most_common_species)

        # Determine the maximum occurrences of the primary species in any single image
        sequence_primary_species_count = max(
            [image_species_list.count(sequence_primary_species) for image_species_list in species_classes if isinstance(image_species_list, list)]
        )


    unique_species = set(species for sublist in species_classes for species in sublist)
    non_interest_species = unique_species - set(SPECIES_OF_INTEREST)

    if not non_interest_species:
        non_interest_species_str = "0"
    else:
        non_interest_species_str = ", ".join(non_interest_species) 

    wild_boar_confidences = [conf for species_list, conf_list in zip(species_classes, species_confidences) for species, conf in zip(species_list, conf_list) if species == "Wild Boar"]
    bear_confidences = [conf for species_list, conf_list in zip(species_classes, species_confidences) for species, conf in zip(species_list, conf_list) if species == "Bear"]
    other_confidences = [conf for species_list, conf_list in zip(species_classes, species_confidences) for species, conf in zip(species_list, conf_list) if species in non_interest_species]

    if wild_boar_confidences:
        min_wild_boar_conf = round(min(wild_boar_confidences) * 100)
        max_wild_boar_conf = round(max(wild_boar_confidences) * 100)
        if min_wild_boar_conf == max_wild_boar_conf:
            wild_boar_confidence_str = f" ({min_wild_boar_conf}%)"
        else:
            wild_boar_confidence_str = f" ({min_wild_boar_conf}-{max_wild_boar_conf}%)"
    else:
        wild_boar_confidence_str = ""

    if bear_confidences:
        min_bear_conf = round(min(bear_confidences) * 100)
        max_bear_conf = round(max(bear_confidences) * 100)
        if min_bear_conf == max_bear_conf:
            bear_confidence_str = f" ({min_bear_conf}%)"
        else:
            bear_confidence_str = f" ({min_bear_conf}-{max_bear_conf}%)"
    else:
        bear_confidence_str = ""

    if other_confidences:
        min_other_conf = round(min(other_confidences) * 100)
        max_other_conf = round(max(other_confidences) * 100)
        if min_other_conf == max_other_conf:
            other_confidence_str = f" ({min_other_conf}%)"
        else:
            other_confidence_str = f" ({min_other_conf}-{max_other_conf}%)"
    else:
        other_confidence_str = ""

    other_count = int(max(animal_count)) - int(max(wild_boar_count)) - int(max(bear_count))


    if priority_alert:
        if priority_alert_count == 1:
            alert_caption = f"🚨<b> {int(priority_alert_count)} {priority_alert.upper()} DETECTED </b>🚨"
        else:
            alert_caption = f"🚨<b> {int(priority_alert_count)} {priority_alert.upper()}S DETECTED </b>🚨"
        print(f"{current_time()} | PRIORITY ALERT: AT LEAST {int(priority_alert_count)} {priority_alert.upper()} DETECTED IN IMAGE SEQUENCE")

        if human_warning:
            alert_caption += f"\n<b>WARNING: {int(max(human_count))} HUMAN(S) and {int(max(vehicle_count))} VEHICLES(S) ALSO DETECTED</b>"

    elif human_warning:
        # Plural handling not added to emphasise human presence in case of human under-detection
        if max(human_count) > 0:
            alert_caption = f"🚶‍➡️<b> {int(max(human_count))} HUMAN(S) DETECTED </b>🚶"
        if max(vehicle_count) > 0:
            alert_caption += "\n" if max(human_count) > 0 else ""
            alert_caption = f"🚜<b> {int(max(vehicle_count))} VEHICLES(S) DETECTED </b>🚜"
        print(f"{current_time()} | WARNING: {int(max(human_count))} HUMAN(S) and {int(max(vehicle_count))} VEHICLES(S) DETECTED IN IMAGE SEQUENCE")
    else:
        if sequence_primary_species_count == 1:
            alert_caption = f"🦊<b> {int(sequence_primary_species_count)} {sequence_primary_species} Detected </b> 🦡"
        else:
            alert_caption = f"🦊 <b> {int(sequence_primary_species_count)} {sequence_primary_species}s Detected </b> 🦡"
        print(f"{current_time()} | {int(sequence_primary_species_count)} Non-Priority Animal(s) ({sequence_primary_species}) Detected")

    alert_caption += f"\n\n🕔 Time: {img_time}"
    alert_caption += f"\n🌍 Location: {location}"
    alert_caption += f"\n📍 <a href='{map_url}'>Map Link</a>"

    if human_warning:
        alert_caption += f"\n\n<i>Do not share photos of humans outside of FCC. No photos of humans are sent between 06:00 and 21:00 to protect privacy. Authorised users can check {EMAIL_USER} to view the photos.</i>"

    alert_caption += f"\n--------------------"
    alert_caption += f"\n🧍 Humans: {int(max(human_count))}"
    alert_caption += f"\n🚜 Vehicles: {int(max(vehicle_count))}"
    alert_caption += f"\n🐗 Wild Boars: {int(max(wild_boar_count))}{wild_boar_confidence_str}"
    alert_caption += f"\n🐻 Bears: {int(max(bear_count))}{bear_confidence_str}"
    alert_caption += f"\n🦊🦌🦡🦉Others: {other_count}{other_confidence_str}"
    alert_caption += f"\n📷 ID: {int(sequence_id)}, {img_date}"

    alert_caption += f"\n--------------------"
    alert_caption += f"\nCamera ID: {camera_id}"
    alert_caption += f"; Brand: {camera_make}"
    alert_caption += f"; Battery: {battery}%"
    alert_caption += f"; Storage: {sd_memory}%"
    if pd.notna(temperature):
        alert_caption += f"; Temperature: {temperature}℃"
    alert_caption += f"; GPS: {gps}"


    flattened_alert_caption = alert_caption.replace('\n', '. ')
    df.iloc[-num_images:, df.columns.get_loc('Alert Message')] = flattened_alert_caption


    return df, alert_caption, priority_alert


from PIL import Image, ImageDraw
import io

def annotate_images(df, images, human_warning, HUMAN_ALERT_START, HUMAN_ALERT_END):
    
    if human_warning:

        current_time_in_romania = get_current_time_in_romania()

        if HUMAN_ALERT_START <= current_time_in_romania or current_time_in_romania < HUMAN_ALERT_END:
            print(f"{current_time()} | Time in Romania: {current_time_in_romania} - sending human photos")
        else:
            print(f"{current_time()} | Time in Romania: {current_time_in_romania} - human photos EXCLUDED from alert")
            return None

    else:
        print(f"{current_time()} | No human warning - sending photos regardless of time")

    image_list = []
    
    # Extract the last len(images) rows
    last_rows = df.iloc[-len(images):]

    for image, row in zip(images, last_rows.iterrows()):
        index, row_data = row
        draw = ImageDraw.Draw(image)

        detection_boxes = row_data['Detection Boxes']
        detection_classes = row_data['Detection Classes']
        detection_confidences = row_data['Detection Confidences']
        species_classes = row_data['Species Classes']
        species_confidences = row_data['Species Confidences']
        
        species_idx = 0  # to keep track of the current species index

        for box, d_class, d_conf in zip(detection_boxes, detection_classes, detection_confidences):
            # Convert relative coordinates to absolute coordinates
            width, height = image.size
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int((box[0] + box[2]) * width)
            y2 = int((box[1] + box[3]) * height)
            
            # Set color and label based on detection class
            if d_class == '1':  # Animal
                color = 'red'
                label = f"{species_classes[species_idx]} {species_confidences[species_idx] * 100:.0f}%"
                species_idx += 1
            elif d_class == '2':  # Human
                color = 'green'
                label = f"Human {d_conf * 100:.0f}%"
            elif d_class == '3':  # Vehicle
                color = 'blue'
                label = f"Vehicle {d_conf * 100:.0f}%"
            else:
                color = 'yellow'
                label = f"Unknown {d_conf * 100:.0f}%"
                
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 10), label, fill=color)
        
        image_list.append(image)
    
    return image_list


import requests
import io
import json
import time
from requests.exceptions import ConnectionError, HTTPError

def send_alert_to_telegram(bot_token, chat_id, photos, caption):
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

    print(f"{current_time()} | Alert sent.")




##################################################
############### HELPER FUNCTIONS #################
##################################################


def detections_in_sequence(df, images):
    last_rows = df.iloc[-len(images):]
    human_or_vehicle_present = (last_rows['Human Count'] > 0).any() or (last_rows['Vehicle Count'] > 0).any()
    primary_species_not_empty = (last_rows['Primary Species'] != "Empty").any()
    return human_or_vehicle_present or primary_species_not_empty


from datetime import datetime
import pytz

def get_current_time_in_romania():
    romania_tz = pytz.timezone('Europe/Bucharest')
    return datetime.now(romania_tz).strftime("%H:%M")

import os
from PIL import Image


def save_images(df, images, human_warning, PHOTOS_PATH):
    # Ensure the PHOTOS_PATH directory exists
    os.makedirs(PHOTOS_PATH, exist_ok=True)

    # Get the last len(images) rows of df
    last_rows = df.tail(len(images))
    
    # Iterate through the images and corresponding rows in the DataFrame
    for index, (image, (_, row)) in enumerate(zip(images, last_rows.iterrows())):
        primary_species = row['Primary Species']
        file_id = row['File ID']

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