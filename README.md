# 🐗🦊 AI for Wildlife Monitoring 🐻🦡
### A Real-Time Alert System Using 4G Camera Traps

**Thomas Ratsakatika | AI and Environment Researcher | University of Cambridge**

This software was developed in partnership with <a href="https://www.carpathia.org/" target="_blank">Fundația Conservation Carpathia</a>, a nature conservation and restoration organisation based in the Romania.

<i>Read the [LICENSE](#-license) and [DISCLAIMER](#-disclaimer) before use.</i>

## 💡 Overview 

This repository contains code to run an wildlife alert system on your laptop, PC or server. You can even run the basic version on a [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (requires 4GB RAM minimum).

The system will process photos sent from a 4G-enabled camera trap and send alerts directly to your mobile phone.

To get started, you will need at least one 4G-enabled camera trap, a dedicated email address, a Telegram account and some basic python skills.

If you don't have these - or you just want to test the system on your laptop - run the [Example Tutorial](#-example-tutorial).

<div style="text-align: center;">
  <img src="assets/example_detections.gif" alt="Annotated Photos"/>
  <p>© Photos credit: <a href="https://www.carpathia.org/" target="_blank">Fundația Conservation Carpathia</a> </p>
</div>

## 📣  Example alert message

<img src="assets/screenshot.png" alt="Example Alert" width="400" />

## 🛠️ Setup

<i>Beta warning: this alert system is still in development and may result in unexpected behaviour. Also note that third-party functionality (e.g. email 'app passwords') may change at any time, resulting in the system not working. As more testing is done, future updates will aim to make the system more robust.</i>

### 🚀 Quick Start (Advanced Users)

Get started by following the steps below. If anything doesn't make sense, follow the more detailed [Example Tutorial](#-example-tutorial).

1. Clone the repository.
2. Download the [detection and classification models](#-models) and move them to the [models](models) directory.
3. Create the camera traps virtual environment with pip (requirement.txt - recommended) or conda (environment.yaml).
4. Create a dedicated email account to receive the 4G camera trap photos and generate an app password. Your email provider **must** support app passwords (e.g. <a href="https://myaccount.google.com/apppasswords" target="_blank">Gmail</a>).
5. Set up your 4G camera trap to send photos to this dedicated email address.
6. Create a bot in Telegram using @BotFather and note down the bot token (<a href="https://core.telegram.org/bots/tutorial" target="_blank">detailed instructions here</a>).
7. Create a group in Telegram, add the bot to the group and make it an admin. Then note down the group's <a href="https://www.wikihow.com/Know-Chat-ID-on-Telegram-on-Android" target="_blank">chat ID</a> (do this AFTER adding the bot - the chat ID should start with '#-100')
8. Update the config.yaml file with your email account settings and Telegram chat ID and bot token.
9. Update the [camera locations](data/camera_locations.csv) CSV file with your camera(s) details, location and a google maps link.
10. Verify that the settings in the the [alert system script](scripts/advanced_alert_system.py) meet your requirements.
11. Activate the virtual environment and run the alert system script: `python3 scripts/advanced_alert_system.py`

The script will check the email account every 60 seconds for unread emails, download any photos, detect and classify animals, and send an alert your Telegram Group. It will then update the [capture database](data/capture_database.csv) and save the original photos in the [photos folder](data/photos). A high level process flow diagramme can be found [here](assets/final_alert_system_flow_diagram.png).

You can also find code for processing camera trap data, and testing and fine tuning AI models in the [archive](archive) folder. These notebooks are experimental and while they contain comments, are no detailed tutorials. [Contact me](#-contact) if you have any queries.

### 🎓 Example Tutorial

To run the [Example Tutorial Notebook](notebooks/alert_system_tutorial.ipynb), you will need to be able to run <a href="https://www.python.org/downloads/" target="_blank">Python</a> in a <a href="https://docs.jupyter.org/en/latest/start/index.html#id1" target="_blank">Jupyter Notebook</a>.

There are many tutorials on how to do this online. One option is to download <a href="https://code.visualstudio.com/" target="_blank">VS Code</a> and add the Python and Jupyter extensions.


Once you are set up with Jupyter, you will need to copy the alert system code repository onto your computer. If you know how to use Git, the command is:

   ```bash
   git clone https://github.com/ratsakatika/camera-traps.git
   ```

VS Code also provides tools to "clone" (copy) a repository. You can alternatively download everything as a zip file by clicking on the green 'code' button at the top of this page.

Now you will need to create a ["virtual environment"](https://docs.python.org/3/library/venv.html) and install all the modules needed to run the alert system. Open a new terminal (within VS Code or your operating system), navigate to the camera-traps folder (`cd camera-traps`), and create a virtual environment with the required modules using pip (recommended) or conda (<a href="https://docs.anaconda.com/miniconda/#" target="_blank">download here</a>):

  - Using pip (Windows):
    ```bash
    python -m venv camera_traps
    camera_traps\Scripts\activate
    pip install -r requirements.txt
    ```

  - Using pip (Linux/macOS):
    ```bash
    python3 -m venv camera_traps
    source camera_traps/bin/activate
    pip install -r requirements.txt
    ```

  - Using conda (Windows/Linux/macOS):
    ```bash
    conda env create -f environment.yml
    conda activate camera_traps
    ```

 You are now ready to open and run the [Example Tutorial Notebook](notebooks/alert_system_tutorial.ipynb).

## 🤖 Models

**Important:** Users must agree to the respective license terms of third-party models. Once downloaded, these models should be moved to [models](models) directory.

The **advanced version** of the alert system uses the <a href="https://github.com/agentmorris/MegaDetector?tab=readme-ov-file" target="_blank">MegaDetector</a> object detection model and <a href="https://www.deepfaune.cnrs.fr/en/" target="_blank">DeepFaune</a> species classification model:

- MegaDetector Detection Model: <a href="https://github.com/agentmorris/MegaDetector/releases/tag/v5.0" target="_blank">md_v5a.0.0.pt</a>
- DeepFaune Classifier Model: <a href="https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1/" target="_blank">deepfaune-vit_large_patch14_dinov2.lvd142m.pt</a>

The **basic version** uses DeepFaune's object detection and species classification models:

- DeepFaune Detection Model: <a href="https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1/" target="_blank">deepfaune-yolov8s_960.pt</a>
- DeepFaune Classifier Model: <a href="https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1/" target="_blank">deepfaune-vit_large_patch14_dinov2.lvd142m.pt</a>

If you are using the alert system in the Carpathian Mountains or would like to bias classification performance towards wild boar and bears, you may also wish to try the fine-tuned DeepFaune classification model, which works alongside MegaDetector. Further details can be found in the accompanying research report [COMING SOON].

- Carpathian Mountain Model: [fine-tuned-deepfaune-balanced-loss.pt](https://drive.google.com/file/d/1Y0P7Z4DX1s7Sj3v7LjbeZNkMC4wByKPi/view?usp=sharing)
- Carpathian Mountain Model (biased to bears and wild boars): [fine-tuned-deepfaune-biased-loss.pt](https://drive.google.com/file/d/1Co5UuHwNKCPfCW3KTZ9M_42HzvS4IpzV/view?usp=sharing)

## 📷 Data

Some example photos are provided in [data/example_photos](data/example_photos) to get you started.

You could also investigate <a href="https://lila.science/" target="_blank">LILA BC</a> and <a href="https://www.wildlifeinsights.org/" target="_blank">WildLife Insights</a> if you want to download large camera trap datasets to train your own model using the notebooks provided in the [archive](archive) folder.

## 📂 Project Structure

```
.
├── archive             # Directory for analysis, fine-tuning and testing notebooks
├── assets              # Directory for images and other media
├── data                # Directory for storing data
│   ├── example_photos  # Subdirectory of example photos
│   └── photos          # Subdirectory for storing photos received
├── models              # Directory for storing the AI models
├── notebooks           # Directory for the example tutorial notebook
├── scripts             # Directory for alert system scripts
├── environment.yml     # Conda environment configuration file
├── requirements.txt    # Pip environment configuration file
├── LICENSE             # License file for the repository
└── README.md           # Readme file with project overview and instructions

```
 
## 📨 Contact

For questions or more information, please contact [Tom Ratsakatika](mailto:trr26@cam.ac.uk).

## 📖 Citation

### Wildlife Alert System (this repository)

```
@misc{ratsakatika2024ai,
  author       = {Thomas Ratsakatika},
  title        = {AI for Wildlife Monitoring},
  year         = 2024,
  month        = jun,
  institution  = {University of Cambridge},
  howpublished = {\url{https://github.com/ratsakatika/camera-traps}}
}
```
### DeepFaune Detection and Classification Models

```
@misc{rigoudy_deepfaune_2024,
  author       = {Noa Rigoudy and Gaspard Dussert and Abdelbaki Benyoub and Aurélien Besnard and Carole Birck},
  title        = {{DeepFaune} / {DeepFaune} {Software} · {GitLab}},
  year         = 2024,
  month        = feb,
  abstract     = {PLMlab - La forge by Mathrice},
  language     = {en},
  howpublished = {\url{https://plmlab.math.cnrs.fr/deepfaune/software}},
  urldate      = {2024-04-08},
  journal      = {GitLab}
}
```

### MegaDetector Detection Model

```
@misc{Beery_Efficient_Pipeline_for,
	title = {Efficient pipeline for camera trap image review},
	copyright = {MIT},
	url = {http://github.com/agentmorris/MegaDetector},
	author = {Beery, Sara and Morris, Dan and Yang, Siyu},
}
```

```
@misc{hernandez2024pytorchwildlife,
  author       = {Andres Hernandez and Zhongqi Miao and Luisa Vargas and Rahul Dodhia and Juan Lavista},
  title        = {Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
  year         = 2024,
  howpublished = {\url{https://github.com/microsoft/CameraTraps/blob/main/megadetector.md}},
  eprint       = {2405.12930},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV}
}
```

## 📜 License

The alert system code is released under the MIT license. Further details can be [found here](LICENSE).

The Carpathian Mountain models are released under the Creative Commons non-Commercial CC BY-NC-SA 4.0 License. Further details can be <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">found here</a>.

The DeepFaune models' license can be <a href="https://plmlab.math.cnrs.fr/deepfaune/software/-/tree/master" target="_blank">found here</a>. Commercial use of the DeepFaune models is forbidden.

The MegaDetector model's license can be <a href="https://github.com/agentmorris/MegaDetector?tab=readme-ov-file" target="_blank">found here</a>.

## ❗ Disclaimer

*Anyone who accesses, uses, or refers to this repository, including but not limited to its code, software, applications, instructions, and any third-party applications or services mentioned or linked herein, takes full responsibility for their actions and outcomes. The author(s) of this repository make no warranties or representations, express or implied, regarding the accuracy, reliability, completeness, or suitability of the code, instructions, or any associated third-party applications or services for any particular purpose. By accessing or using this repository, you agree to do so at your own risk and in accordance with the terms of the relevant licenses. The author(s) shall not be held liable for any damages, including but not limited to direct, indirect, incidental, or consequential damages, arising from the use or misuse of the code, applications, instructions, or any third-party applications or services. Users are advised to review and comply with all applicable laws, licenses, and terms of use of any third-party code, applications, or services referenced in this repository.*

## 🙏 Acknowledgements

I would like to extend my sincere thanks to <a href="https://www.cst.cam.ac.uk/people/sk818" target="_blank">Professor Srinivasan Keshav</a> and <a href="https://www.researchgate.net/profile/Ruben-Iosif" target="_blank">Dr Ruben Iosif</a> for their invaluable insights and expertise while developing this software. I would also like to thank Fundația Conservation Carpathia's Wildlife and Rapid Intervention Team, who hosted me in Romania, deployed the camera traps, and provided valuable feedback during the system's development.

This software was built as part of the UKRI-funded [Artificial Intelligence for Environmental Risk](https://ai4er-cdt.esc.cam.ac.uk/) Centre for Doctoral Training programme at the University Cambridge.

<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <td align="center">
        <img src="assets/logo_cambridge_colour.jpg" alt="University of Cambridge" width="600" />
      </td>
      <td align="center">
        <img src="assets/logo_ukri_colour.png" alt="UKRI Logo" width="600" />
      </td>
    </tr>
  </table>
</div>
