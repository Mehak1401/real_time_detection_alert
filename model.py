import cv2
import numpy as np
import os
import time
import threading
import queue
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import serial
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Input
from sklearn.ensemble import IsolationForest

# Download NLTK resources
nltk.download('vader_lexicon')

class SurveillanceSystem:
    def __init__(self, video_source=0, arduino_port=None, model_path=None, 
                 alert_email=None, alert_threshold=0.7):
        """
        Initialize the surveillance system.
        
        Args:
            video_source: Camera index or video file path
            arduino_port: Serial port for Arduino communication
            model_path: Path to pre-trained anomaly detection model
            alert_email: Email to send alerts to
            alert_threshold: Threshold for anomaly detection (0.0 to 1.0)
        """
        self.video_source = video_source
        self.arduino_port = arduino_port
        self.alert_email = alert_email
        self.alert_threshold = alert_threshold
        
        # Initialize video capture
        self.cap = None
        
        # Initialize queues for processing
        self.frame_queue = queue.Queue(maxsize=30)
        self.audio_queue = queue.Queue(maxsize=10)
        self.alert_queue = queue.Queue()
        
        # Initialize variables for anomaly detection
        self.motion_history = []
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        
        # Load or create anomaly detection model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self._build_anomaly_detection_model()
            print("New anomaly detection model created")
        
        # Isolation Forest for detecting anomalies
        self.isolation_forest = IsolationForest(contamination=0.01, random_state=42)
            
        # Initialize Arduino connection
        self.arduino = None
        if arduino_port:
            try:
                self.arduino = serial.Serial(arduino_port, 9600, timeout=1)
                print(f"Connected to Arduino on {arduino_port}")
            except:
                print(f"Failed to connect to Arduino on {arduino_port}")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Flags
        self.running = False
        self.alert_mode = False
        
        # Statistics
        self.stats = {
            "processed_frames": 0,
            "anomalies_detected": 0,
            "alerts_sent": 0
        }
    
    def _build_anomaly_detection_model(self):
        """Build a model combining MobileNetV2 features with LSTM for temporal analysis"""
        # Base feature extractor
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Model architecture
        input_layer = Input(shape=(10, 224, 224, 3))  # 10 frames sequence
        
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(128)(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def start(self):
        """Start the surveillance system"""
        if self.running:
            print("System is already running")
            return
            
        self.running = True
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            self.running = False
            return
            
        # Start worker threads
        self.threads = []
        
        # Video processing thread
        video_thread = threading.Thread(target=self._process_video_stream)
        video_thread.daemon = True
        self.threads.append(video_thread)
        
        # Audio processing thread
        audio_thread = threading.Thread(target=self._process_audio_stream)
        audio_thread.daemon = True
        self.threads.append(audio_thread)
        
        # Arduino data thread
        if self.arduino:
            arduino_thread = threading.Thread(target=self._process_arduino_data)
            arduino_thread.daemon = True
            self.threads.append(arduino_thread)
        
        # Alert handling thread
        alert_thread = threading.Thread(target=self._handle_alerts)
        alert_thread.daemon = True
        self.threads.append(alert_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
            
        print("Surveillance system started")
    
    def stop(self):
        """Stop the surveillance system"""
        self.running = False
        
        # Release video capture
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        # Close Arduino connection
        if self.arduino:
            self.arduino.close()
            
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        print("Surveillance system stopped")
        
    def _process_video_stream(self):
        """Process video frames for anomaly detection"""
        frame_buffer = []
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
                
            # Resize for processing
            processed_frame = cv2.resize(frame, (224, 224))
            
            # Store original frame for display/alerts
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame_data = {
                "frame": frame.copy(),
                "processed": processed_frame,
                "timestamp": timestamp,
                "frame_id": frame_count
            }
            
            # Add to frame queue for display/saving
            if not self.frame_queue.full():
                self.frame_queue.put(frame_data)
            
            # Apply background subtraction for motion detection
            fg_mask = self.background_subtractor.apply(processed_frame)
            motion_level = np.sum(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1] * 255)
            
            # Add frame to buffer for feature-based analysis
            frame_buffer.append(processed_frame)
            if len(frame_buffer) > 10:
                frame_buffer.pop(0)
            
            # Update motion history
            self.motion_history.append(motion_level)
            if len(self.motion_history) > 100:  # Keep last 100 frames
                self.motion_history.pop(0)
            
            # Detect anomalies if we have enough frames
            if len(frame_buffer) == 10 and frame_count % 5 == 0:  # Check every 5 frames
                # Prepare batch for model
                batch = np.array([frame_buffer])
                batch = preprocess_input(batch)
                
                # Get model prediction
                anomaly_score = self.model.predict(batch, verbose=0)[0][0]
                
                # Get motion-based features
                if len(self.motion_history) >= 30:
                    recent_motion = np.array(self.motion_history[-30:]).reshape(1, -1)
                    if frame_count % 100 == 0:  # Retrain isolation forest periodically
                        self.isolation_forest.fit(np.array(self.motion_history).reshape(-1, 1))
                    
                    # Get isolation forest anomaly score (-1 for anomalies, 1 for normal)
                    motion_anomaly = self.isolation_forest.predict(recent_motion)[0]
                    motion_anomaly_score = 0.5 if motion_anomaly == 1 else 0.9
                    
                    # Combine scores (weighted average)
                    combined_score = 0.7 * anomaly_score + 0.3 * motion_anomaly_score
                    
                    # If anomaly detected
                    if combined_score > self.alert_threshold:
                        self.stats["anomalies_detected"] += 1
                        alert_data = {
                            "type": "video_anomaly",
                            "score": combined_score,
                            "frame": frame.copy(),
                            "timestamp": timestamp,
                            "sensor_data": None
                        }
                        self.alert_queue.put(alert_data)
                        print(f"Anomaly detected! Score: {combined_score:.2f}")
            
            # Update stats
            self.stats["processed_frames"] += 1
            frame_count += 1
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    def _process_audio_stream(self):
        """Process audio stream for speech-to-text and sentiment analysis"""
        # This assumes you have a microphone connected
        # In a real implementation, you might need to sync this with video
        
        with sr.Microphone() as source:
            print("Audio processing started")
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.running:
                try:
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Convert speech to text
                    text = self.recognizer.recognize_google(audio)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    negative_score = sentiment['neg']
                    
                    print(f"Speech detected: '{text}'")
                    print(f"Sentiment: {sentiment}")
                    
                    # If negative sentiment detected, raise alert
                    if negative_score > 0.5:
                        alert_data = {
                            "type": "audio_anomaly",
                            "speech": text,
                            "sentiment": sentiment,
                            "timestamp": timestamp,
                            "sensor_data": None
                        }
                        self.alert_queue.put(alert_data)
                        print(f"Negative speech detected! Score: {negative_score:.2f}")
                        
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Error in audio processing: {e}")
                    
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
    
    def _process_arduino_data(self):
        """Process data from Arduino (GPS and temperature)"""
        while self.running and self.arduino:
            try:
                if self.arduino.in_waiting > 0:
                    data = self.arduino.readline().decode('utf-8').strip()
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Parse Arduino data (expected format: "GPS:lat,long;TEMP:value")
                    try:
                        parts = data.split(';')
                        sensor_data = {}
                        
                        for part in parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'GPS':
                                    lat, lng = value.split(',')
                                    sensor_data['gps'] = {
                                        'latitude': float(lat),
                                        'longitude': float(lng)
                                    }
                                elif key == 'TEMP':
                                    sensor_data['temperature'] = float(value)
                        
                        print(f"Arduino data: {sensor_data}")
                        
                        # Check for temperature anomalies (example: over 40°C might indicate fire)
                        if 'temperature' in sensor_data and sensor_data['temperature'] > 40:
                            alert_data = {
                                "type": "temperature_anomaly",
                                "temperature": sensor_data['temperature'],
                                "timestamp": timestamp,
                                "sensor_data": sensor_data
                            }
                            self.alert_queue.put(alert_data)
                            print(f"High temperature detected: {sensor_data['temperature']}°C")
                            
                    except Exception as e:
                        print(f"Error parsing Arduino data: {e}")
                        
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                
            # Small delay to prevent CPU overuse
            time.sleep(0.1)
    
    def _handle_alerts(self):
        """Process and handle alerts"""
        while self.running:
            try:
                # Check if there's an alert in the queue
                if not self.alert_queue.empty():
                    alert_data = self.alert_queue.get()
                    
                    # Print alert information
                    print(f"\nALERT - {alert_data['timestamp']}")
                    print(f"Type: {alert_data['type']}")
                    
                    # Save frame if available
                    if 'frame' in alert_data:
                        alert_frame = alert_data['frame']
                        filename = f"alert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, alert_frame)
                        print(f"Alert image saved as {filename}")
                    
                    # Get sensor data if available
                    sensor_info = ""
                    if alert_data['sensor_data']:
                        sensor_info = f"Sensor Data: {alert_data['sensor_data']}"
                        print(sensor_info)
                    
                    # Send email alert if configured
                    if self.alert_email:
                        self._send_email_alert(alert_data)
                        
                    self.stats["alerts_sent"] += 1
                    self.alert_queue.task_done()
                    
            except Exception as e:
                print(f"Error handling alert: {e}")
                
            # Small delay to prevent CPU overuse
            time.sleep(0.1)
    
    def _send_email_alert(self, alert_data):
        """Send email alert with image if available"""
        try:
            # Configure email
            msg = MIMEMultipart()
            msg['From'] = 'surveillance.system@example.com'
            msg['To'] = self.alert_email
            msg['Subject'] = f"ALERT: {alert_data['type']} - {alert_data['timestamp']}"
            
            # Email body
            body = f"Alert Type: {alert_data['type']}\n"
            body += f"Timestamp: {alert_data['timestamp']}\n"
            
            if alert_data['type'] == 'video_anomaly':
                body += f"Anomaly Score: {alert_data['score']:.2f}\n"
            elif alert_data['type'] == 'audio_anomaly':
                body += f"Speech: {alert_data['speech']}\n"
                body += f"Sentiment: {alert_data['sentiment']}\n"
            elif alert_data['type'] == 'temperature_anomaly':
                body += f"Temperature: {alert_data['temperature']}°C\n"
                
            if alert_data['sensor_data']:
                body += f"\nSensor Data:\n"
                if 'gps' in alert_data['sensor_data']:
                    gps = alert_data['sensor_data']['gps']
                    body += f"GPS: {gps['latitude']}, {gps['longitude']}\n"
                    body += f"Google Maps: https://maps.google.com/?q={gps['latitude']},{gps['longitude']}\n"
                if 'temperature' in alert_data['sensor_data']:
                    body += f"Temperature: {alert_data['sensor_data']['temperature']}°C\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if available
            if 'frame' in alert_data:
                img_data = cv2.imencode('.jpg', alert_data['frame'])[1].tobytes()
                image = MIMEImage(img_data, name="alert_image.jpg")
                msg.attach(image)
            
            # Send email - this is a placeholder, you'll need to configure SMTP settings
            # with smtplib.SMTP('smtp.example.com', 587) as server:
            #     server.starttls()
            #     server.login('username', 'password')
            #     server.send_message(msg)
            
            print(f"Alert email sent to {self.alert_email}")
            
        except Exception as e:
            print(f"Error sending email alert: {e}")
    
    def get_stats(self):
        """Get system statistics"""
        return self.stats
    
    def display_feed(self):
        """Display the video feed with anomaly indicators"""
        while self.running:
            if not self.frame_queue.empty():
                frame_data = self.frame_queue.get()
                frame = frame_data["frame"]
                timestamp = frame_data["timestamp"]
                
                # Add timestamp to frame
                cv2.putText(frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add stats to frame
                cv2.putText(frame, f"Frames: {self.stats['processed_frames']}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Anomalies: {self.stats['anomalies_detected']}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Surveillance Feed", frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
            else:
                time.sleep(0.01)
        
        cv2.destroyAllWindows()

    def train_model(self, normal_footage_path, anomaly_footage_path=None, epochs=10):
        """
        Train the anomaly detection model using normal and anomalous footage.
        
        Args:
            normal_footage_path: Directory containing normal footage videos
            anomaly_footage_path: Directory containing anomalous footage (optional)
            epochs: Number of training epochs
        """
        # This is a simplified training function - in a real implementation,
        # you would need more sophisticated data loading and preprocessing
        
        print("Training anomaly detection model...")
        print("This would involve:")
        print("1. Loading and preprocessing video data")
        print("2. Extracting frame sequences")
        print("3. Training the model with normal vs anomalous data")
        print("4. Saving the trained model")
        print("5. Evaluating performance")
        
        # In a real implementation, you would:
        # 1. Create a data generator that reads videos
        # 2. Extract frame sequences and labels (normal=0, anomaly=1)
        # 3. Train the model using model.fit()
        # 4. Save the model using model.save()
        
        print(f"Training would run for {epochs} epochs")
        print("Note: Actual implementation requires substantial data preparation")

# Usage example
if __name__ == "__main__":
    # Create surveillance system
    system = SurveillanceSystem(
        video_source=0,  # Use camera index 0 (webcam)
        arduino_port="COM3",  # Change to your Arduino port
        alert_email="alerts@example.com"  # Change to your email
    )
    
    try:
        # Start the system
        system.start()
        
        # Display video feed in main thread
        system.display_feed()
        
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        system.stop()
        print("System stopped.")