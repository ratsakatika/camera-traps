import yaml
import time
from datetime import datetime
from megadetector.detection import run_detector
import pandas as pd

# Adjust the path as needed to point to the directory containing alert_system_utils.py
import sys
sys.path.append('../scripts') 

from alert_system_utils import (
    detector,
    classifier,
    set_device,
    check_emails,
    extract_and_update_camera_info,
    update_camera_data_dataframe,
    batch_classification,
    current_time,
    generate_alert_caption,
    detections_in_sequence,
    send_alert_to_telegram,
    save_images,
    annotate_images,
    get_romanian_class
)

# Load settings from configuration file
with open('../config.yaml') as file:
    config = yaml.safe_load(file)

IMAP_HOST = config['imap_config']['host']
EMAIL_USER = config['imap_config']['user']
EMAIL_PASS = config['imap_config']['password']
TELEGRAM_BOT_TOKEN = config['telegram_config']['bot_token']
TELEGRAM_CHAT_ID = config['telegram_config']['chat_id']  # '-1002249589791' # config['telegram_config']['chat_id']  #  replace with config after tests # 

# Detection and Classification Model Settings
DETECTOR_MODEL_PATH = '../models/md_v5a.0.0.pt'
DETECTOR_CLASSES = ["animal", "human", "vehicle"]
DETECTION_THRESHOLD = 0.10

BACKBONE = 'vit_large_patch14_dinov2'
CLASSIFIER_MODEL_PATH = '../models/deepfaune-vit_large_patch14_dinov2.lvd142m.pt'
CLASSIFIER_CLASSES = [
    "Badger", "Ibex", "Red Deer", "Chamois", "Cat",
    "Goat", "Roe Deer", "Dog", "Squirrel", "Equid", "Genet",
    "Hedgehog", "Lagomorph", "Wolf", "Lynx", "Marmot",
    "Micromammal", "Mouflon", "Sheep", "Mustelid", "Bird",
    "Bear", "Nutria", "Fox", "Wild Boar", "Cow"
]
ROMANIAN_CLASSES = [
    "Bursuc", "Ibex", "Cerb", "Capră Neagră", "Pisică", 
    "Capră", "Caprior(ară)", "Câine", "Veveriț", "Cal", "Genetă",
    "Arici", "Lagomorf", "Lup", "Râs", "Marmotă", 
    "Micromamifer", "Muflon", "Oaie", "Mustelid", "Pasăre", 
    "Urs", "Nutrie", "Vulpe", "Mistret", "Vacă"
]
SPECIES_OF_INTEREST = ["Wild Boar", "Bear"]
CLASSIFICATION_THRESHOLD = 0.20

CAPTURE_DATABASE_PATH = '../data/capture_database.csv'
CAMERA_LOCATIONS_PATH = '../data/camera_locations.csv'
PHOTOS_PATH = '../data/photos/'
ALERT_LANGUAGE = "ro" # Enter 'en' for English, 'ro' for Romanian
HUMAN_ALERT_START = "21:00"
HUMAN_ALERT_END = "06:00"
CHECK_EMAIL_FREQUENCY = 60

# Initialise the Detection and Classifier Models
device = set_device()
detector_model = run_detector.load_detector(DETECTOR_MODEL_PATH)
print("Loading classifier...")
start_time = time.time()
classifier_model = classifier(CLASSIFIER_MODEL_PATH, BACKBONE, CLASSIFIER_CLASSES, device)
end_time = time.time()
print(f"Loaded classifier in {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
    while True:
        try:
            images, original_images, camera_id, temp_deg_c, img_date, img_time, battery, sd_memory = \
                check_emails(IMAP_HOST, EMAIL_USER, EMAIL_PASS)
            if camera_id:
                camera_make, gps, location, map_url, battery, sd_memory = extract_and_update_camera_info(CAMERA_LOCATIONS_PATH, camera_id, battery, sd_memory)
                if images:
                    df = pd.read_csv(CAPTURE_DATABASE_PATH)
                    df = update_camera_data_dataframe(df, len(images), camera_id, camera_make, img_date, img_time, temp_deg_c, battery, sd_memory, location, gps, map_url)
                    df, human_warning = detector(df, detector_model, images, DETECTION_THRESHOLD)
                    df = batch_classification(df, classifier_model, images, CLASSIFICATION_THRESHOLD)
                    if detections_in_sequence(df, images):
                        df, alert_caption = generate_alert_caption(df, human_warning, HUMAN_ALERT_START, HUMAN_ALERT_END, len(images), SPECIES_OF_INTEREST, EMAIL_USER, ALERT_LANGUAGE, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)                        
                        alert_images = annotate_images(df, images, human_warning, HUMAN_ALERT_START, HUMAN_ALERT_END, ALERT_LANGUAGE, CLASSIFIER_CLASSES, ROMANIAN_CLASSES)
                        send_alert_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alert_images, alert_caption)
                    else:
                        print(f"{current_time()} | All photos in sequence are empty")
                    df = save_images(df, original_images, human_warning, PHOTOS_PATH)
                    df.to_csv(CAPTURE_DATABASE_PATH, index=False)
                    print(f"{current_time()} | Capture database updated: {CAPTURE_DATABASE_PATH}")
                    del df
                else:
                    print(f"{current_time()} | No images attached to email")
                print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
            else:
                time.sleep(CHECK_EMAIL_FREQUENCY)
        except KeyboardInterrupt:
            print(f"{current_time()} | Interrupted by user")
            break

        except Exception as e:
            print(f"{current_time()} | An error occurred: {e}")
            time.sleep(CHECK_EMAIL_FREQUENCY)
            print(f"\n{current_time()} | Monitoring {EMAIL_USER} for new messages...")
            continue

