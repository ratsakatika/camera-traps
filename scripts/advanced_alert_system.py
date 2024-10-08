################################################
########### IMPORT REQUIRED MODULES ############
################################################

from megadetector.detection import run_detector  # MegaDetector object detection model
import pandas as pd             # Data manipulation and analysis
import yaml                     # Handles YAML (config) files
import time                     # Time functions
from datetime import datetime   # Date and time manipulation
import schedule                 # Schedule function for weekly reports
import functools                # Function for weekly reports
import sys                      # System parameters and functions
sys.path.append('../scripts')   # Add scripts directory to system path

### Import alert system functions from ../scripts/alert_system_utils.py ###

from alert_system_utils import (

    ### Functions to download photos and metadata from emails ###

    current_time,               # Get the current time
    check_emails,               # Checks for new emails, extracts photos and metadata
    extract_and_update_camera_info,  # Extract and update camera information
    update_camera_data_dataframe,    # Update camera data DataFrame

    ### Functions to detect and classify animals in photos ###

    set_device,                 # Sets computation device (CPU/GPU)
    detector,                   # Animal/human/vehicle detection
    classifier,                 # Animal classification model
    batch_classification,       # Batch classification of images
    detections_in_sequence,     # Checks if anything has been detected

    ### Functions to annotate photos and send an alert to Telegram ###

    generate_alert_caption,     # Generate captions for alerts
    send_alert_to_telegram,     # Send alerts to Telegram
    annotate_images,            # Annotate images with detection results

    ### Functions to save the photos and send weekly reports ###
    save_images,                # Save images to disk
    send_weekly_report          # Send a weekly report
)

###############################################################
####### INITIALISE DETECTION AND CLASSIFICATION MODELS ########
###############################################################

DETECTION_THRESHOLD = 0.15      # Detection threshold - recommended 0.15
CLASSIFICATION_THRESHOLD = 0.20 # Classification threshold - recommend set between 0.2 to 0.8 depending on tolerance for false positives

# Alert Message Settings
ALERT_LANGUAGE = "en"           # Curently English (en) and Romanian (ro) are supported
HUMAN_ALERT_START = "21:00"     # Privacy feature: start/end time that photos of people may be sent
HUMAN_ALERT_END = "06:00"

# Detection Model Settings
DETECTOR_MODEL_PATH = '../models/md_v5a.0.0.pt'
DETECTOR_CLASSES = ["animal", "human", "vehicle"]

# Classification Model Settings
BACKBONE = 'vit_large_patch14_dinov2'
CLASSIFIER_MODEL_PATH = '../models/deepfaune-vit_large_patch14_dinov2.lvd142m.pt'   # Change to fine-tuned model if desired
CLASSIFIER_CLASSES = [
    "Badger", "Ibex", "Red Deer", "Chamois", "Cat",
    "Goat", "Roe Deer", "Dog", "Squirrel", "Equid", "Genet",
    "Hedgehog", "Lagomorph", "Wolf", "Lynx", "Marmot",
    "Micromammal", "Mouflon", "Sheep", "Mustelid", "Bird",
    "Bear", "Nutria", "Fox", "Wild Boar", "Cow"
]
ROMANIAN_CLASSES = [
    "Bursuc", "Ibex", "Cerb", "Capră Neagră", "Pisică", 
    "Capră", "Căprioară", "Câine", "Veveriță", "Cal", "Genetă",
    "Arici", "Iepuri", "Lup", "Râs", "Marmotă", 
    "Micromamifer", "Muflon", "Oaie", "Mustelid", "Pasăre", 
    "Urs", "Nutrie", "Vulpe", "Mistreț", "Vacă"
]
SPECIES_OF_INTEREST = ["Wild Boar", "Bear"] # Species for which priority alerts are sent (currently only supports wild boar and bears)

# Locations of capture database, camera location tables, and storage folder for photos received by the alert system
CAPTURE_DATABASE_PATH = '../data/capture_database.csv' 
CAMERA_LOCATIONS_PATH = '../data/camera_locations.csv'
PHOTOS_PATH = '../data/photos/'

# Initialise the Detection and Classifier Models
device = set_device()
detector_model = run_detector.load_detector(DETECTOR_MODEL_PATH)
print("Loading classifier...")
start_time = time.time()
classifier_model = classifier(CLASSIFIER_MODEL_PATH, BACKBONE, CLASSIFIER_CLASSES, device)
end_time = time.time()
print(f"Loaded classifier in {(end_time - start_time):.2f} seconds")

#################################################
####### LOAD EMAIL AND TELEGRAM SETTINGS ########
#################################################

# Load settings from configuration file
with open('../config.yaml') as file:
    config = yaml.safe_load(file)

IMAP_HOST = config['imap_config']['host']
EMAIL_USER = config['imap_config']['user']
EMAIL_PASS = config['imap_config']['password']

TELEGRAM_BOT_TOKEN = config['telegram_config']['bot_token']
TELEGRAM_CHAT_ID =  config['telegram_config']['chat_id']

SMTP_SERVER = config['smtp_config']['host']
SMTP_PORT = int(config['smtp_config']['port'])
EMAIL_SENDER = config['smtp_config']['user']
EMAIL_PASSWORD = config['smtp_config']['password']
RECIPIENTS = config['smtp_config']['recipients']

CHECK_EMAIL_FREQUENCY = 60      # Sets how often the system checks for emails (default - 60 seconds)

### Settings for regular report (default - weekly on Mondays at 08:00)
schedule.every().monday.at("08:00").do(
    functools.partial(send_weekly_report, SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, SMTP_PORT, CAPTURE_DATABASE_PATH, CAMERA_LOCATIONS_PATH, RECIPIENTS, EMAIL_USER)
)

################################################
########### START ALERT SYSTEM LOOP ############
################################################

if __name__ == "__main__":
    print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
    while True:
        try:
            # Check for emails, extract metadata
            images, original_images, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory = \
                check_emails(IMAP_HOST, EMAIL_USER, EMAIL_PASS)
            
            if camera_id:
                
                # Get the camera details from the camera_location databse
                camera_make, gps, location, map_url, battery, sd_memory = extract_and_update_camera_info(CAMERA_LOCATIONS_PATH, camera_id, battery, sd_memory)
                
                if images:
                    
                    # If images are attached to the email, open the capture_database
                    df = pd.read_csv(CAPTURE_DATABASE_PATH)
                    
                    # Update the capture database with rows for each image
                    df = update_camera_data_dataframe(df, len(images), camera_id, camera_make, img_date, img_time, temp_deg_c, battery, sd_memory, location, gps, map_url)
                    
                    # Run the detection model on each image
                    df, human_warning = detector(df, detector_model, images, DETECTION_THRESHOLD)
                    
                    # Run the classification model on each image
                    df = batch_classification(df, classifier_model, images, CLASSIFICATION_THRESHOLD)
                    
                    if detections_in_sequence(df, images):
                        
                        # If there are any detections above the threshold, create an alert caption
                        df, alert_caption = generate_alert_caption(df, human_warning, HUMAN_ALERT_START, HUMAN_ALERT_END, len(images), SPECIES_OF_INTEREST, EMAIL_USER, ALERT_LANGUAGE, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)                        
                        
                        # If no humans/vehicles are detected, annotate the photos with boxes and species labels
                        alert_images = annotate_images(df, images, human_warning, HUMAN_ALERT_START, HUMAN_ALERT_END, ALERT_LANGUAGE, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)
                        
                        # Send an alert to the Telegram group
                        send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alert_images, alert_caption)
                    
                    else:
                        
                        print(f"{current_time()} | All photos in sequence are empty")
                    
                    # Save the original photos the the ../data/photos/ folder
                    df = save_images(df, original_images, human_warning, PHOTOS_PATH)
                    
                    # Update the capture_database
                    df.to_csv(CAPTURE_DATABASE_PATH, index=False)
                    print(f"{current_time()} | Capture database updated: {CAPTURE_DATABASE_PATH}")
                    
                    # Clear the dataframe to free up memory
                    del df
                
                else:
                    
                    print(f"{current_time()} | No images attached to email")
                
                print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
            
            else:
                # Wait before checking emails again
                time.sleep(CHECK_EMAIL_FREQUENCY)
            
            # Check to see if it is time to send the regular report
            schedule.run_pending()
        
        except KeyboardInterrupt:
            print(f"{current_time()} | Interrupted by user")
            break
        
        # Error handling to keep the system running if an error occurs
        except Exception as e:
            print(f"{current_time()} | An error occurred: {e}")
            time.sleep(CHECK_EMAIL_FREQUENCY)
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
            continue