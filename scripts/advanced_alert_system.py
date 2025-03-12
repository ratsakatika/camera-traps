### Import required modules ###

from PytorchWildlife.models import detection as pw_detection # MegaDetector object detection model

import pandas as pd             # Data manipulation and analysis
import time                     # Time functions
from datetime import datetime   # Date and time manipulation
import schedule                 # Schedule function for weekly reports
import functools                # Function for weekly reports
import sys                      # System parameters and functions
import os                       # Functions for interacting with the operating system
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

    generate_alert_caption_en,     # Generate captions for alerts (English)
    generate_alert_caption_ro,     # Generate captions for alerts (Romanian)
    send_alert_to_telegram,        # Send alerts to Telegram
    annotate_images_en,            # Annotate images with detection results (English)
    annotate_images_ro,            # Annotate images with detection results (Romanian)

    ### Functions to save the photos and send weekly reports ###
    save_images,                   # Save images to disk
    send_weekly_report,            # Send a weekly report

    #### Function to load the settings from the configuration file
    load_config                    # Loads configuration file settings
)


###############################################################
####### INITIALISE DETECTION AND CLASSIFICATION MODELS ########
###############################################################

# Set Device - GPU or CPU
device = set_device()

# Detection Model Settings

DETECTOR_MODEL_V6 = False   # True = v6, False = v5
DETECTOR_CLASSES = ["animal", "human", "vehicle"]

print(f"Loading detector...")
if DETECTOR_MODEL_V6:
    weights = os.path.join("../models/MDV6-yolov10x.pt")
    detector_model = pw_detection.MegaDetectorV6(device=device, weights=weights, pretrained=True, version="MDV6-yolov10-e")
    detector_model.predictor.args.verbose = False
    print("Using MegaDetector v6 (MDV6-yolov9-e)")
else:
    weights = os.path.join("../models/md_v5a.0.0.pt")
    detector_model = pw_detection.MegaDetectorV5(device=device, weights=weights, pretrained=True, version="a")
    print("Using MegaDetector v5 (md_v5a.0.0)")

# Classification Model Settings
BACKBONE = 'vit_large_patch14_dinov2'

BACKBONE = 'vit_large_patch14_dinov2'
CLASSIFIER_MODEL_PATH = '../models/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt'  # Update to the new model path if needed

CLASSIFIER_CLASSES = [
    "Bison", "Badger", "Ibex", "Beaver", "Red Deer", "Chamois", "Cat",
    "Goat", "Roe Deer", "Dog", "Fallow Deer", "Squirrel", "Moose",
    "Equid", "Genet", "Wolverine", "Hedgehog", "Lagomorph", "Wolf",
    "Otter", "Lynx", "Marmot", "Micromammal", "Mouflon", "Sheep",
    "Mustelid", "Bird", "Bear", "Nutria", "Raccoon", "Fox",
    "Reindeer", "Wild Boar", "Cow"
]

ROMANIAN_CLASSES = [
    "Bizon", "Bursuc", "Ibex", "Castor", "Cerb", "Capră Neagră", "Pisică",
    "Capră", "Căprioară", "Câine", "Cerb Lopătar", "Veveriță", "Elan",
    "Cal", "Genetă", "Jder Mare", "Arici", "Iepure", "Lup",
    "Vidră", "Râs", "Marmotă", "Micromamifer", "Muflon", "Oaie",
    "Mustelid", "Pasăre", "Urs", "Nutrie", "Raton", "Vulpe",
    "Ren", "Mistreț", "Vacă"
]

print(f"\nLoading classifier...")
start_time = time.time()
classifier_model = classifier(CLASSIFIER_MODEL_PATH, BACKBONE, CLASSIFIER_CLASSES, device)
end_time = time.time()
print(f"Loaded classifier in {(end_time - start_time):.2f} seconds. Device: {device}")


#################################################
###### INITIALISE SETTINGS AND FILE PATHS #######
#################################################

# Locations of capture database, camera location tables, and storage folder for photos received by the alert system
CAPTURE_DATABASE_PATH = '../data/capture_database.csv' 
CAMERA_LOCATIONS_PATH = '../data/camera_locations.csv'
PHOTOS_PATH = '../data/photos/'

# Load configuration from file
CONFIG_PATH = '../config.yaml'
config = load_config(CONFIG_PATH)

# Extract the variables from the returned dictionary
DETECTION_THRESHOLD = config['DETECTION_THRESHOLD']
DETECTOR_MODEL_V6 = config['DETECTOR_MODEL_V6']
CLASSIFICATION_THRESHOLD = config['CLASSIFICATION_THRESHOLD']
ALERT_LANGUAGE = config['ALERT_LANGUAGE']
PRIORITY_SPECIES = config['PRIORITY_SPECIES']
CHECK_EMAIL_FREQUENCY = config['CHECK_EMAIL_FREQUENCY']
IMAP_HOST = config['IMAP_HOST']
EMAIL_USER = config['EMAIL_USER']
EMAIL_PASS = config['EMAIL_PASS']
TELEGRAM_BOT_TOKEN = config['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID_ALL = config['TELEGRAM_CHAT_ID_ALL']
TELEGRAM_CHAT_ID_PRIORITY = config['TELEGRAM_CHAT_ID_PRIORITY']
TELEGRAM_CHAT_ID_HUMAN = config['TELEGRAM_CHAT_ID_HUMAN']
SMTP_SERVER = config['SMTP_SERVER']
SMTP_PORT = config['SMTP_PORT']
EMAIL_SENDER = config['EMAIL_SENDER']
EMAIL_PASSWORD = config['EMAIL_PASSWORD']
RECIPIENTS = config['RECIPIENTS']
last_config_mod_time = config['last_config_mod_time']

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
            images, original_images, image_count, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory = \
                check_emails(IMAP_HOST, EMAIL_USER, EMAIL_PASS)

            if camera_id:

                # Get the camera details from the camera_location database
                camera_make, gps, location, map_url, battery, sd_memory = extract_and_update_camera_info(CAMERA_LOCATIONS_PATH, camera_id, battery, sd_memory)
                
                if images:
                    
                    # If images are attached to the email, open the capture_database
                    df = pd.read_csv(CAPTURE_DATABASE_PATH)

                    # Update the capture database with rows for each image
                    df = update_camera_data_dataframe(df, image_count, camera_id, camera_make, img_date, img_time, temp_deg_c, battery, sd_memory, location, gps, map_url)
                    
                    # Run the detection model on each image
                    df, human_warning = detector(df, detector_model, images, DETECTION_THRESHOLD)
                    
                    # Run the classification model on each image
                    df, priority_detections = batch_classification(df, classifier_model, images, CLASSIFICATION_THRESHOLD, PRIORITY_SPECIES)
                    
                    # Checks if there are any animal detections > CLASSIFICATION_THRESHOLD or human/vehicle detections of any confidence level
                    if detections_in_sequence(df, images, CLASSIFICATION_THRESHOLD):

                        # Generate an alert in English or Romanian
                        if ALERT_LANGUAGE == "ro":

                            df, alert_caption = generate_alert_caption_ro(df, human_warning, image_count, priority_detections, CLASSIFICATION_THRESHOLD, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)                        
                            alert_images = annotate_images_ro(df, images, CLASSIFICATION_THRESHOLD, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)

                        else: # Default to English

                            df, alert_caption = generate_alert_caption_en(df, human_warning, image_count, priority_detections, CLASSIFICATION_THRESHOLD)
                            alert_images = annotate_images_en(df, images, CLASSIFICATION_THRESHOLD)

                        if human_warning:
                            send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_HUMAN, alert_images, alert_caption)
                            print(f"{current_time()} | Alert sent to people/vehicle group.")
                            
                            if TELEGRAM_CHAT_ID_ALL != TELEGRAM_CHAT_ID_HUMAN:
                                send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_ALL, None, alert_caption)
                                print(f"{current_time()} | Alert sent to all animals group (photos withheld).")
                        
                        elif priority_detections:
                            send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_PRIORITY, alert_images, alert_caption)
                            print(f"{current_time()} | Alert sent to priority animals group")
                            
                            if TELEGRAM_CHAT_ID_ALL != TELEGRAM_CHAT_ID_PRIORITY:
                                send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_ALL, alert_images, alert_caption)
                                print(f"{current_time()} | Alert sent to all animals group")
                        
                        else:
                            send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID_ALL, alert_images, alert_caption)
                            print(f"{current_time()} | Alert sent to all animals group")
                            
                    else:
                        
                        print(f"{current_time()} | All photos in sequence are empty")
                    
                    # Save the original photos the the ../data/photos/ folder
                    df = save_images(df, original_images, human_warning, PHOTOS_PATH, CLASSIFICATION_THRESHOLD)
                    
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

            # Check for updates to the config file
            current_config_mod_time = os.path.getmtime(CONFIG_PATH)

            if current_config_mod_time != last_config_mod_time:
                print(f"{current_time()} | Configuration File Updated")
                print(f"{current_time()} | Original Settings: "
                    f"DETECTION_THRESHOLD={DETECTION_THRESHOLD}, CLASSIFICATION_THRESHOLD={CLASSIFICATION_THRESHOLD}, "
                    f"ALERT_LANGUAGE={ALERT_LANGUAGE}, CHECK_EMAIL_FREQUENCY={CHECK_EMAIL_FREQUENCY} seconds, "
                    f"PRIORITY_SPECIES={PRIORITY_SPECIES}")
                load_config(CONFIG_PATH)
                last_config_mod_time = current_config_mod_time

        except KeyboardInterrupt:
            print(f"{current_time()} | Interrupted by user")
            break
        
        # Error handling to keep the system running if an error occurs
        except Exception as e:
            print(f"{current_time()} | An error occurred: {e}")
            time.sleep(CHECK_EMAIL_FREQUENCY)
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
            continue