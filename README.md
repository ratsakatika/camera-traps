# 🐻 AI for Wildlife Monitoring 🐗
### A Real-Time Alert System Using 4G Camera Traps

**Tom Ratsakatika | AI and Environment Researcher | University of Cambridge**

<i>**Read the [LICENSE](#-license) and [DISCLAIMER](#-disclaimer) before use.**</i>

## 💡 Overview 

This repository contains code to run an wildlife alert system on your computer.

The system will process photos sent from a 4G-enabled camera trap and send alerts directly to your mobile phone.

To get started, you will need at lest one 4G-enabled camera trap, a dedicated email address, a Telegram account and some basic python skills.

If you don't have all of these, you can still test out the system in the [Example Tutorial](#example-tutorial).

<div style="text-align: center;">
  <img src="assets/example_detections.gif" alt="Annotated Photos"/>
  <p>Photo credits: <a href="https://www.carpathia.org/">Fundația Conservation Carpathia</a> ©</p>
</div>

## 📣  Example alert message

<img src="assets/screenshot.png" alt="Example Alert" width="400" />

## 🚀 Get Started

### Quick Start (Advanced Users)

Get started by following the 10 steps below. If anything doesn't make sense, follow the more detailed [Example Tutorial](#example-tutorial).

1. Clone the repository
2. Download the [detection and classification models](#-models).
2. Create the camera-traps virtual environment with pip (requirement.txt - recommended) or conda (environment.yaml).
3. Create a dedicated email account to receive camera trap photos and generate an app password. Your email provider **must** support app passwords (e.g. [Gmail](https://myaccount.google.com/apppasswords)).
4. Set up your 4G-enabled camera trap to send photos to this dedicated email address.
5. Create a group in Telegram and note down the [chat ID](https://www.wikihow.com/Know-Chat-ID-on-Telegram-on-Android) (it should start with '#-100').
6. Create a bot in Telegram using @BotFather and note down the bot token ([detailed instructions here](https://core.telegram.org/bots/tutorial)).
7. Update the config.yaml file with your email account settings and Telegram chat ID and bot token.
8. Update the [camera locations](data/camera_locations.csv) CSV file with your camera(s) details, location and a google maps link.
9. Verify that the settings in the the [alert system script](scripts/advanced_alert_system.py) meet your requirements.
10. Activate the virtual environment and run the alert system script: `python3 python3 scripts/advanced_alert_system.py`

Now, every photo 

### Example Tutorial

To run the [Example Tutorial](notebooks/alert_system_tutorial.ipynb), you'll need to be able to run [Python](https://www.python.org/downloads/) in a [Jupyter Notebook](https://docs.jupyter.org/en/latest/start/index.html#id1).

There are many tutorials on how to do this online. One option is to download [VS Code](https://code.visualstudio.com/) and add the Python and Jupyter extensions.

Once you are set up with Jupyter, you will need to copy the alert system code repository onto your computer. If you know how to use Git, the command is:

   ```bash
   git clone https://github.com/ratsakatika/camera-traps.git
   ```

VS Code also provides tools to "clone" (copy) a repository. You can also simply download everything as a zip file by clicking on the code button above.

Now you'll need to create a "virtual environment", which will contain all the programmes needed to run the alert system. Open a new terminal (within VS Code or your operating system), navigate to the camera-traps folder (`cd camera-traps`), and create an virtual environment using pip (recommended) or conda ([download here](https://docs.anaconda.com/miniconda/#)):

  - Using pip (Windows):
    ```bash
    python -m venv camera_traps
    source camera_traps\Scripts\activate
    pip install -r requirements.txt
    ```

  - Using pip (Linux):
    ```bash
    python -m venv camera_traps
    source camera_traps/bin/activate
    pip install -r requirements.txt
    ```

   - Using conda (Windows/Linux):
     ```bash
     conda env create -f environment.yaml
     conda activate camera_traps
     ```


 You're now ready to open and run the [Example Tutorial Notebook](notebooks/alert_system_tutorial.ipynb).


## 📂 Project Structure

### 🛠️ Setup

### 🤖 Models

The pretrained DeepFaune models can be downloaded here: https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1/

### 📷 Data
Example photos


## 📧 Contact

Tom Ratsakatika: trr26@cam.ac.uk

Citation

## 📜 License

MIT License

## ❗ Disclaimer

*Anyone who accesses, uses, or refers to this repository, including but not limited to its code, instructions, and any third-party applications or services mentioned or linked herein, takes full responsibility for their actions and outcomes. The author(s) of this repository make no warranties or representations, express or implied, regarding the accuracy, reliability, completeness, or suitability of the code, instructions, or any associated third-party applications or services for any particular purpose. By accessing or using this repository, you agree to do so at your own risk and in accordance with the terms of the LICENSE. The author(s) shall not be held liable for any damages, including but not limited to direct, indirect, incidental, or consequential damages, arising from the use or misuse of the code, instructions, or any third-party applications or services. Users are advised to review and comply with all applicable laws, licenses, and terms of use of any third-party code, applications, or services referenced in this repository.*

## 🙏 Acknowledgements

<table>
  <tr align="center">
    <td align="center">
      <img src="assets/logo_cambridge_colour.jpg" alt="University of Cambridge" width="300" />
    </td>
    <td align="center">
      <img src="assets/logo_ukri_colour.png" alt="UKRI Logo" width="300" />
    </td>
  </tr>
</table>
