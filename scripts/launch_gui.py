import sys
import yaml
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QFileDialog, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex

CONFIG_FILE = "../config.yaml"
SCRIPT_FILE = "advanced_alert_system.py"

class LogReaderThread(QThread):
    log_updated = pyqtSignal(list)

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.running = True
        self.log_buffer = []
        self.mutex = QMutex()

    def run(self):
        while self.running:
            if self.process.poll() is not None:
                break  # Process has terminated
            output = self.process.stdout.readline().strip()
            if output:
                self.mutex.lock()
                self.log_buffer.append(output)
                self.mutex.unlock()
        self.process.stdout.close()

    def get_logs(self):
        self.mutex.lock()
        logs = self.log_buffer[:]
        self.log_buffer.clear()
        self.mutex.unlock()
        return logs

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class ConfigEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Configuration")
        self.setGeometry(350, 100, 1200, 800)  # Increased default window size
        
        self.process = None  # Stores the subprocess running the system
        self.log_thread = None
        self.settings_changed = False

        self.layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        
        self.config_data = self.load_config()
        self.fields = {}
        
        for section, values in self.config_data.items():
            for key, value in values.items():
                if isinstance(value, list):  # Handle list fields (like recipients)
                    field = QLineEdit(", ".join(map(str, value)))  # Comma-separated for lists
                else:
                    field = QLineEdit(str(value))
                field.textChanged.connect(self.enable_update_button)
                self.fields[f"{section}.{key}"] = field
                self.form_layout.addRow(QLabel(f"{section}.{key}"), field)
        
        self.layout.addLayout(self.form_layout)
        
        self.update_button = QPushButton("Update Settings")
        self.update_button.setEnabled(False)  # Initially disabled
        self.update_button.clicked.connect(self.save_config)
        self.layout.addWidget(self.update_button)
        
        self.start_button = QPushButton("Start System")
        self.start_button.clicked.connect(self.toggle_system)
        self.layout.addWidget(self.start_button)
        
        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(250)  # Ensure at least 10 lines of logs are visible
        self.layout.addWidget(QLabel("System Logs:"))
        self.layout.addWidget(self.log_list)
        
        self.setLayout(self.layout)
        
        # Timer to update logs efficiently
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.flush_log_buffer)
        self.log_timer.start(200)  # Update logs every 200ms

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                return yaml.safe_load(file) or {}
        return {}
    
    def save_config(self):
        for key, field in self.fields.items():
            section, option = key.split('.')
            if isinstance(self.config_data[section].get(option, None), list):  # Convert back to list using commas
                self.config_data[section][option] = [email.strip() for email in field.text().split(",") if email.strip()]
            else:
                self.config_data[section][option] = field.text().strip()
        
        with open(CONFIG_FILE, 'w') as file:
            yaml.dump(self.config_data, file, default_flow_style=False)
        print("Configuration updated!")
        self.update_button.setEnabled(False)  # Disable button after saving
    
    def enable_update_button(self):
        self.update_button.setEnabled(True)
    
    def toggle_system(self):
        if self.process is None:
            self.start_system()
        else:
            self.stop_system()
    
    def start_system(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output

        self.process = subprocess.Popen(
            [sys.executable, SCRIPT_FILE], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True, 
            env=env
        )
        
        self.log_thread = LogReaderThread(self.process)
        self.log_thread.start()
        self.start_button.setText("Stop System")
        print("System started.")
    
    def stop_system(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.log_thread.stop()
            self.log_thread = None
            self.start_button.setText("Start System")
            print("System stopped.")
    
    def flush_log_buffer(self):
        if self.log_thread:
            logs = self.log_thread.get_logs()
            for message in logs:
                self.log_list.addItem(message)
                if self.log_list.count() > 1000:
                    self.log_list.takeItem(0)  # Remove the oldest log entry
            self.log_list.scrollToBottom()
    
    def closeEvent(self, event):
        # Ensure the process is stopped if the user closes the GUI
        self.stop_system()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigEditor()
    window.show()
    sys.exit(app.exec())
